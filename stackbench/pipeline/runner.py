"""
Pipeline runner that orchestrates the complete documentation validation workflow.

This module coordinates:
1. Repository cloning
2. Documentation extraction
3. API signature validation
4. Code example validation
"""

import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from opentelemetry.trace import Status, StatusCode
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console

from stackbench.repository import RepositoryManager, RunContext
from stackbench.cache import CacheManager
from stackbench.telemetry import RunLogger, get_tracer
from stackbench.agents import (
    DocumentationExtractionAgent,
    APISignatureValidationAgent,
    CodeExampleValidationAgent,
    DocumentationClarityAgent,
    ExtractionSummary,
)

console = Console()


class DocumentationValidationPipeline:
    """Orchestrates the complete documentation validation pipeline."""

    def __init__(
        self,
        repo_url: str,
        branch: str,
        library_name: str,
        library_version: str,
        base_output_dir: Path,
        commit: Optional[str] = None,
        docs_path: Optional[str] = None,
        include_folders: Optional[List[str]] = None,
        num_workers: int = 5,
    ):
        """
        Initialize the validation pipeline.

        Args:
            repo_url: Git repository URL
            branch: Git branch to clone
            library_name: Primary library being documented
            library_version: Library version to validate against
            base_output_dir: Base directory for all outputs
            commit: Optional commit hash (if None, resolves from branch HEAD)
            docs_path: Base documentation directory (e.g., 'docs/src')
            include_folders: Folders relative to docs_path to analyze
            num_workers: Number of parallel workers for extraction (default: 5)
        """
        self.repo_url = repo_url
        self.branch = branch
        self.commit = commit
        self.docs_path = docs_path
        self.library_name = library_name
        self.library_version = library_version
        self.base_output_dir = Path(base_output_dir)
        self.include_folders = include_folders
        self.num_workers = num_workers

        # Generate unique run ID
        self.run_id = str(uuid.uuid4())

        run_metadata = {
            "repo_url": repo_url,
            "branch": branch,
            "docs_path": docs_path,
            "include_folders": include_folders or [],
            "library": library_name,
            "library_version": library_version,
            "num_workers": num_workers,
        }

        run_dir = self.base_output_dir / self.run_id
        self.run_logger = RunLogger(run_id=self.run_id, run_dir=run_dir, metadata=run_metadata)
        self.tracer = get_tracer(__name__)

        # Initialize repository manager and cache manager
        self.repo_manager = RepositoryManager(base_data_dir=self.base_output_dir)
        self.cache_manager = CacheManager(data_dir=self.base_output_dir)

        # Will be set during execution
        self.run_context: Optional[RunContext] = None
        self.docs_folder: Optional[Path] = None

    async def clone_repository(self) -> RunContext:
        """
        Clone the repository and set up the run directory structure.

        Returns:
            RunContext with cloned repository
        """
        print(f"\n🔄 Cloning repository: {self.repo_url}")
        print(f"   Branch: {self.branch}")
        if self.commit:
            print(f"   Commit: {self.commit}")
        print(f"   Run ID: {self.run_id}")

        self.run_context = self.repo_manager.clone_repository(
            repo_url=self.repo_url,
            branch=self.branch,
            run_id=self.run_id,
            library_name=self.library_name,
            library_version=self.library_version,
            commit=self.commit,
            docs_path=self.docs_path,
            include_folders=self.include_folders
        )

        print(f"✅ Repository cloned to: {self.run_context.repo_dir}")

        # Find documentation files
        md_result = self.repo_manager.find_markdown_files(
            self.run_context,
            include_folders=self.include_folders
        )
        md_files = md_result['files']

        # Store document discovery metrics in context
        self.run_context.total_markdown_files = md_result['total_found']
        self.run_context.markdown_in_include_folders = md_result['in_include_folders']
        self.run_context.filtered_api_reference_count = md_result['filtered_api_reference']
        self.run_context.validated_document_count = len(md_files)
        self.run_context.save_metadata()

        # Display document discovery stats
        print(f"\n📄 Document Discovery:")
        print(f"   Total .md/.mdx files in repo: {md_result['total_found']}")
        if self.include_folders:
            folders_display = ', '.join(self.include_folders)
            print(f"   In include folders ({folders_display}): {md_result['in_include_folders']}")
        print(f"   Filtered (API reference): {md_result['filtered_api_reference']}")
        print(f"   Will validate: {len(md_files)}")

        # Set docs folder based on include_folders
        if self.include_folders and len(self.include_folders) == 1:
            # Single folder specified - use it directly
            self.docs_folder = self.run_context.repo_dir / self.include_folders[0]
        else:
            # Multiple folders or none - use repo root
            self.docs_folder = self.run_context.repo_dir

        # Create output directories
        (self.run_context.results_dir / "extraction").mkdir(exist_ok=True)
        (self.run_context.results_dir / "api_validation").mkdir(exist_ok=True)
        (self.run_context.results_dir / "code_validation").mkdir(exist_ok=True)
        (self.run_context.results_dir / "clarity_validation").mkdir(exist_ok=True)

        # Create validation tracking log directory
        (self.run_context.run_dir / "validation_logs").mkdir(exist_ok=True)

        return self.run_context


    async def _estimate_and_sort_documents(self, md_files: List[Path]) -> List[Path]:
        """
        Estimate document processing time and sort longest-first.

        Uses file size as a proxy for processing time (larger docs take longer).

        Args:
            md_files: List of markdown file paths

        Returns:
            List of paths sorted by size (largest first)
        """
        doc_sizes = []
        for doc in md_files:
            size = doc.stat().st_size
            doc_sizes.append((doc, size))

        # Sort by size descending (largest first)
        doc_sizes.sort(key=lambda x: x[1], reverse=True)

        sorted_docs = [doc for doc, size in doc_sizes]

        print(f"\n📊 Document processing order (by size):")
        for i, (doc, size) in enumerate(doc_sizes[:10], 1):  # Show first 10
            kb = size / 1024
            print(f"   {i}. {doc.name:40s} ({kb:7.1f} KB)")
        if len(doc_sizes) > 10:
            print(f"   ... and {len(doc_sizes) - 10} more")

        return sorted_docs

    async def _process_document_end_to_end(self, doc_file: Path) -> Dict[str, Any]:
        """
        Run complete validation pipeline for a single document.

        Executes all four stages (extraction, API validation, code validation, clarity)
        for one document. If extraction fails, downstream stages are skipped.

        Args:
            doc_file: Path to markdown file to process

        Returns:
            Dict with results from all stages
        """
        if not self.run_context:
            raise RuntimeError("Run context must be initialized before processing documents")

        doc_name = doc_file.stem
        start_time = datetime.now()

        extraction_dir = self.run_context.results_dir / "extraction"
        api_output_dir = self.run_context.results_dir / "api_validation"
        code_output_dir = self.run_context.results_dir / "code_validation"
        clarity_output_dir = self.run_context.results_dir / "clarity_validation"
        validation_log_dir = self.run_context.run_dir / "validation_logs"
        repo_dir = self.run_context.repo_dir

        document_attributes = {
            "document.name": doc_name,
            "document.path": str(doc_file),
            "run.id": self.run_id,
        }

        with self.tracer.start_as_current_span("stackbench.document", attributes=document_attributes) as document_span:
            api_result = None
            code_result = None
            clarity_result = None

            try:
                # 1. EXTRACTION
                extraction_agent = DocumentationExtractionAgent(
                    docs_folder=self.docs_folder,
                    output_folder=extraction_dir,
                    repo_root=repo_dir,
                    default_version=self.library_version,
                    num_workers=1,
                    validation_log_dir=validation_log_dir
                )

                extraction_step_id = await self.run_logger.start_step(
                    name="Extraction",
                    stage="extraction",
                    document=doc_name,
                    metadata={"path": str(doc_file)}
                )

                try:
                    with self.tracer.start_as_current_span(
                        "stackbench.document.extraction",
                        attributes={"document.name": doc_name}
                    ):
                        extraction_result = await extraction_agent.process_document(
                            doc_file,
                            library_name=self.library_name,
                            run_logger=self.run_logger,
                            run_step_id=extraction_step_id
                        )
                except Exception as exc:
                    await self.run_logger.end_step(extraction_step_id, status="failed", error=str(exc))
                    document_span.record_exception(exc)
                    document_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                else:
                    extraction_status = "success" if extraction_result else "failed"
                    extraction_error = None if extraction_result else "Extraction produced no result"
                    await self.run_logger.end_step(extraction_step_id, status=extraction_status, error=extraction_error)

                    if not extraction_result:
                        document_span.set_status(Status(StatusCode.ERROR, "Extraction failed"))
                        return {
                            "document": doc_name,
                            "status": "failed",
                            "stage_failed": "extraction",
                            "extraction": None,
                            "api_validation": {"status": "skipped", "reason": "extraction failed"},
                            "code_validation": {"status": "skipped", "reason": "extraction failed"},
                            "clarity_validation": {"status": "skipped", "reason": "extraction failed"}
                        }

                extraction_file = extraction_dir / f"{doc_name}_analysis.json"

                # 2. API VALIDATION
                api_step_id = await self.run_logger.start_step(
                    name="API Validation",
                    stage="api_validation",
                    document=doc_name,
                    metadata={"extraction_file": str(extraction_file)}
                )

                api_agent = APISignatureValidationAgent(
                    extraction_folder=extraction_dir,
                    output_folder=api_output_dir,
                    num_workers=1,
                    validation_log_dir=validation_log_dir
                )

                try:
                    with self.tracer.start_as_current_span(
                        "stackbench.document.api_validation",
                        attributes={"document.name": doc_name}
                    ):
                        api_result = await api_agent.validate_document(
                            extraction_file,
                            run_logger=self.run_logger,
                            run_step_id=api_step_id
                        )
                except Exception as exc:
                    await self.run_logger.end_step(api_step_id, status="failed", error=str(exc))
                    document_span.record_exception(exc)
                    document_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                else:
                    api_status = "success" if api_result else "failed"
                    await self.run_logger.end_step(api_step_id, status=api_status, error=None if api_status == "success" else "API validation produced no result")

                # 3. CODE VALIDATION
                code_step_id = await self.run_logger.start_step(
                    name="Code Validation",
                    stage="code_validation",
                    document=doc_name,
                    metadata={"extraction_file": str(extraction_file)}
                )

                code_agent = CodeExampleValidationAgent(
                    extraction_output_folder=extraction_dir,
                    validation_output_folder=code_output_dir,
                    num_workers=1,
                    validation_log_dir=validation_log_dir
                )

                try:
                    with self.tracer.start_as_current_span(
                        "stackbench.document.code_validation",
                        attributes={"document.name": doc_name}
                    ):
                        code_result = await code_agent.validate_document(
                            extraction_file,
                            run_logger=self.run_logger,
                            run_step_id=code_step_id
                        )
                except Exception as exc:
                    await self.run_logger.end_step(code_step_id, status="failed", error=str(exc))
                    document_span.record_exception(exc)
                    document_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                else:
                    code_status = "success" if code_result else "failed"
                    await self.run_logger.end_step(code_step_id, status=code_status, error=None if code_status == "success" else "Code validation produced no result")

                # 4. CLARITY VALIDATION
                clarity_step_id = await self.run_logger.start_step(
                    name="Clarity Validation",
                    stage="clarity_validation",
                    document=doc_name,
                    metadata={"extraction_file": str(extraction_file)}
                )

                clarity_agent = DocumentationClarityAgent(
                    extraction_folder=extraction_dir,
                    output_folder=clarity_output_dir,
                    repository_folder=repo_dir,
                    num_workers=1,
                    validation_log_dir=validation_log_dir
                )

                try:
                    with self.tracer.start_as_current_span(
                        "stackbench.document.clarity_validation",
                        attributes={"document.name": doc_name}
                    ):
                        clarity_result = await clarity_agent.analyze_document(
                            extraction_file,
                            run_logger=self.run_logger,
                            run_step_id=clarity_step_id
                        )
                except Exception as exc:
                    await self.run_logger.end_step(clarity_step_id, status="failed", error=str(exc))
                    document_span.record_exception(exc)
                    document_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                else:
                    clarity_status = "success" if clarity_result else "failed"
                    await self.run_logger.end_step(clarity_step_id, status=clarity_status, error=None if clarity_status == "success" else "Clarity validation produced no result")

                duration = (datetime.now() - start_time).total_seconds()
                document_span.set_attribute("document.duration_seconds", duration)
                document_span.set_status(Status(StatusCode.OK))

                return {
                    "document": doc_name,
                    "status": "success",
                    "duration_seconds": duration,
                    "extraction": extraction_result,
                    "api_validation": api_result,
                    "code_validation": code_result,
                    "clarity_validation": clarity_result
                }

            except Exception as exc:
                document_span.record_exception(exc)
                document_span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    async def _worker_with_progress(
        self,
        worker_id: int,
        document_queue: asyncio.Queue,
        progress: Progress,
        overall_task
    ) -> List[Dict[str, Any]]:
        """
        Worker that processes documents from queue with progress updates.

        Each worker takes documents from the shared queue and runs the full
        pipeline (extract → API → code → clarity) for each document.

        Args:
            worker_id: Unique worker identifier (0 to num_workers-1)
            document_queue: Shared queue of documents to process
            progress: Rich Progress instance for updates
            overall_task: Progress task ID for overall completion

        Returns:
            List of results for all documents processed by this worker
        """
        worker_results = []

        while True:
            try:
                # Get next document (non-blocking with timeout)
                doc_file = await asyncio.wait_for(
                    document_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                # Queue is empty
                break

            try:
                # Run full pipeline for this document (silently)
                result = await self._process_document_end_to_end(doc_file)

                worker_results.append(result)

                # Update progress
                progress.update(overall_task, advance=1)

            except Exception as e:
                console.print(f"[red]❌ Worker {worker_id}: Failed {doc_file.name}: {e}[/red]")
                import traceback
                traceback.print_exc()

                worker_results.append({
                    "document": doc_file.stem,
                    "status": "failed",
                    "error": str(e)
                })

                # Update progress even on failure
                progress.update(overall_task, advance=1)

            finally:
                document_queue.task_done()

        return worker_results

    async def run(self, force: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline using worker pool pattern.

        Each worker processes documents end-to-end (extract → API → code → clarity).
        Documents are sorted longest-first to minimize idle time at end.

        Args:
            force: If True, bypass cache and re-run analysis

        Returns:
            Dict with all summaries
        """
        overall_start = datetime.now()

        if not force:
            resolved_commit = self.repo_manager.resolve_commit_hash(
                self.repo_url, self.branch, self.commit
            )

            console.print(f"\n🔍 Checking cache for commit {resolved_commit}...")

            cached_run_id = self.cache_manager.get_cached_run(
                repo_url=self.repo_url,
                doc_commit_hash=resolved_commit,
                docs_path=self.docs_path,
                library_name=self.library_name,
                library_version=self.library_version
            )

            if cached_run_id:
                console.print(f"\n✅ [bold green]Cache hit![/bold green] Using results from run: {cached_run_id}")
                console.print(f"   Cached run directory: {self.base_output_dir / cached_run_id}")
                console.print(f"   [dim]Use --force to bypass cache and re-run analysis[/dim]")

                await self.run_logger.start_run({
                    "cache_hit": True,
                    "resolved_commit": resolved_commit,
                })
                await self.run_logger.log_info(
                    "Cache hit - using existing results",
                    {"cached_run_id": cached_run_id}
                )
                await self.run_logger.complete_run(status="cached")

                cached_run_dir = self.base_output_dir / cached_run_id
                return {
                    "run_id": cached_run_id,
                    "status": "cached",
                    "cached": True,
                    "cache_hit": True,
                    "run_dir": str(cached_run_dir),
                    "message": "Results loaded from cache"
                }

            console.print("   Cache miss - running new analysis")

        else:
            console.print("\n⚡ Force mode enabled - bypassing cache")

        run_started = False
        run_status = "completed"
        run_error: Optional[str] = None
        result: Dict[str, Any] = {}

        try:
            await self.run_logger.start_run()
            run_started = True

            run_attributes = {
                "run.id": self.run_id,
                "repo.url": self.repo_url,
                "repo.branch": self.branch,
                "docs.path": self.docs_path or "",
                "library.name": self.library_name,
                "library.version": self.library_version,
                "pipeline.workers": self.num_workers,
                "pipeline.force": force,
            }

            with self.tracer.start_as_current_span("stackbench.run", attributes=run_attributes) as run_span:
                try:
                    console.print("\n[cyan]🔄 Cloning repository...[/cyan]")
                    with self.tracer.start_as_current_span("stackbench.clone_repository"):
                        await self.clone_repository()

                    if self.run_context and self.run_context.doc_commit_hash:
                        self.cache_manager.add_run(
                            run_id=self.run_id,
                            repo_url=self.repo_url,
                            branch=self.branch,
                            doc_commit_hash=self.run_context.doc_commit_hash,
                            docs_path=self.docs_path,
                            include_folders=self.include_folders or [],
                            library_name=self.library_name,
                            library_version=self.library_version,
                            status="in_progress"
                        )
                        await self.run_logger.log_info(
                            "Repository cloned",
                            {
                                "repo_dir": str(self.run_context.repo_dir),
                                "doc_commit_hash": self.run_context.doc_commit_hash,
                            },
                        )
                        run_span.set_attribute("repo.commit", self.run_context.doc_commit_hash)

                    md_result = self.repo_manager.find_markdown_files(
                        self.run_context,
                        include_folders=self.include_folders
                    )
                    md_files = md_result['files']

                    await self.run_logger.log_info(
                        "Markdown discovery completed",
                        {
                            "total_found": md_result['total_found'],
                            "in_scope": md_result['in_include_folders'],
                            "filtered_api_reference": md_result['filtered_api_reference'],
                            "to_validate": len(md_files),
                        }
                    )

                    if not md_files:
                        message = "No markdown files found"
                        console.print("[red]❌ No markdown files found[/red]")
                        run_status = "failed"
                        run_error = message
                        await self.run_logger.log_error(message, {"phase": "discovery"})
                        run_span.set_status(Status(StatusCode.ERROR, message))
                        if self.run_context:
                            self.cache_manager.update_run_status(self.run_id, "failed")
                        return {
                            "run_id": self.run_id,
                            "status": "failed",
                            "reason": "no markdown files found"
                        }

                    sorted_docs = await self._estimate_and_sort_documents(md_files)
                    await self.run_logger.log_info(
                        "Document processing order established",
                        {
                            "total_documents": len(sorted_docs),
                            "preview": [doc.name for doc in sorted_docs[:5]],
                        }
                    )

                    document_queue = asyncio.Queue()
                    for doc in sorted_docs:
                        await document_queue.put(doc)

                    console.print(f"\n[bold cyan]🚀 Launching {self.num_workers} workers to process {len(sorted_docs)} documents[/bold cyan]\n")
                    await self.run_logger.log_info(
                        "Workers launched",
                        {"num_workers": self.num_workers, "documents": len(sorted_docs)}
                    )

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console,
                    ) as progress:

                        overall_task = progress.add_task(
                            f"[cyan]Processing {len(sorted_docs)} documents",
                            total=len(sorted_docs)
                        )

                        workers = [
                            self._worker_with_progress(worker_id, document_queue, progress, overall_task)
                            for worker_id in range(self.num_workers)
                        ]

                        worker_results = await asyncio.gather(*workers)

                    all_results = [r for worker in worker_results for r in worker]

                    overall_duration = (datetime.now() - overall_start).total_seconds()

                    console.print(f"\n[bold green]✨ Pipeline complete in {overall_duration:.1f}s ({overall_duration/60:.1f}m)[/bold green]")
                    console.print(f"   Documents processed: [green]{len([r for r in all_results if r['status'] == 'success'])}[/green]/{len(all_results)}")
                    console.print(f"   Failed: [red]{len([r for r in all_results if r['status'] == 'failed'])}[/red]")

                    if self.run_context:
                        self.run_context.mark_analysis_completed()
                        self.cache_manager.update_run_status(self.run_id, "completed")

                    await self.run_logger.log_info(
                        "Pipeline summary",
                        {
                            "duration_seconds": overall_duration,
                            "documents_total": len(all_results),
                            "documents_successful": len([r for r in all_results if r['status'] == 'success']),
                            "documents_failed": len([r for r in all_results if r['status'] == 'failed']),
                        }
                    )

                    run_span.set_attribute("results.documents", len(all_results))
                    run_span.set_attribute("results.successful", len([r for r in all_results if r['status'] == 'success']))
                    run_span.set_attribute("results.failed", len([r for r in all_results if r['status'] == 'failed']))
                    run_span.set_status(Status(StatusCode.OK))

                    result = {
                        "run_id": self.run_id,
                        "duration_seconds": overall_duration,
                        "num_workers": self.num_workers,
                        "total_documents": len(all_results),
                        "successful": len([r for r in all_results if r['status'] == 'success']),
                        "failed": len([r for r in all_results if r['status'] == 'failed']),
                        "results": all_results,
                        "cached": False,
                        "cache_hit": False
                    }

                except Exception as exc:
                    run_status = "failed"
                    run_error = str(exc)
                    run_span.record_exception(exc)
                    run_span.set_status(Status(StatusCode.ERROR, run_error))
                    await self.run_logger.log_error(run_error, {"phase": "pipeline"})
                    if self.run_context:
                        self.cache_manager.update_run_status(self.run_id, "failed")
                    raise

        finally:
            if run_started:
                await self.run_logger.complete_run(status=run_status, error=run_error)

        return result
