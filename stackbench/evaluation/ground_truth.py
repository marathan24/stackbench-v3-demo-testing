"""
Ground truth dataset management for evaluation validation.

Manages gold standard datasets with human annotations for validating
LLM-as-judge evaluation quality. Addresses Mistake #1 from EVALUATION_ANALYSIS.md:
No calibration against human judgment.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class HumanAnnotation(BaseModel):
    """Human annotation for a document or issue."""

    document_id: str = Field(description="Unique document identifier")
    annotator_id: str = Field(description="ID of human annotator")
    annotated_at: str = Field(description="ISO timestamp of annotation")

    # Quality scores (0-10 scale, same as LLM)
    overall_clarity: float = Field(ge=0, le=10)
    instruction_clarity: float = Field(ge=0, le=10)
    logical_flow: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    consistency: float = Field(ge=0, le=10)
    prerequisite_coverage: float = Field(ge=0, le=10)

    # Issues identified
    issues_found: List[Dict[str, any]] = Field(default_factory=list)

    # Metadata
    experience_level: str = Field(description="Annotator experience: beginner/intermediate/expert")
    time_spent_minutes: Optional[int] = Field(None, description="Time spent on annotation")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Annotator confidence in their judgment")
    notes: Optional[str] = Field(None)


class GroundTruthDataset(BaseModel):
    """Collection of human-annotated documents (gold standard)."""

    dataset_id: str
    created_at: str
    description: str
    documents: List[HumanAnnotation]

    # Metadata
    num_annotators_per_doc: int = Field(default=1, description="How many people annotated each doc")
    inter_annotator_agreement: Optional[float] = Field(None, description="Krippendorff's alpha or Cohen's kappa")


class GroundTruthManager:
    """Manages ground truth datasets for evaluation validation."""

    def __init__(self, dataset_dir: Path):
        """
        Initialize manager.

        Args:
            dataset_dir: Directory to store ground truth datasets
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self,
        dataset_id: str,
        description: str,
        num_annotators_per_doc: int = 3
    ) -> GroundTruthDataset:
        """
        Create a new ground truth dataset.

        Args:
            dataset_id: Unique dataset identifier
            description: Description of dataset purpose
            num_annotators_per_doc: Number of annotators per document (default: 3)

        Returns:
            Empty GroundTruthDataset

        Example:
            ```python
            manager = GroundTruthManager(Path("data/ground_truth"))
            dataset = manager.create_dataset(
                dataset_id="lancedb_docs_v1",
                description="30 LanceDB docs annotated by 3 experts",
                num_annotators_per_doc=3
            )
            ```
        """
        dataset = GroundTruthDataset(
            dataset_id=dataset_id,
            created_at=datetime.utcnow().isoformat() + 'Z',
            description=description,
            documents=[],
            num_annotators_per_doc=num_annotators_per_doc
        )

        return dataset

    def add_annotation(
        self,
        dataset: GroundTruthDataset,
        annotation: HumanAnnotation
    ) -> GroundTruthDataset:
        """
        Add a human annotation to the dataset.

        Args:
            dataset: GroundTruthDataset to add to
            annotation: HumanAnnotation to add

        Returns:
            Updated dataset
        """
        dataset.documents.append(annotation)
        return dataset

    def save_dataset(self, dataset: GroundTruthDataset) -> Path:
        """
        Save dataset to disk.

        Args:
            dataset: GroundTruthDataset to save

        Returns:
            Path to saved file
        """
        file_path = self.dataset_dir / f"{dataset.dataset_id}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(dataset.model_dump_json(indent=2))

        return file_path

    def load_dataset(self, dataset_id: str) -> Optional[GroundTruthDataset]:
        """
        Load dataset from disk.

        Args:
            dataset_id: Dataset ID to load

        Returns:
            GroundTruthDataset or None if not found
        """
        file_path = self.dataset_dir / f"{dataset_id}.json"

        if not file_path.exists():
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return GroundTruthDataset(**data)

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List of dataset IDs
        """
        return [
            f.stem for f in self.dataset_dir.glob("*.json")
        ]

    def get_annotations_for_document(
        self,
        dataset: GroundTruthDataset,
        document_id: str
    ) -> List[HumanAnnotation]:
        """
        Get all annotations for a specific document.

        Args:
            dataset: GroundTruthDataset
            document_id: Document identifier

        Returns:
            List of annotations (may be multiple if multi-annotator)
        """
        return [
            ann for ann in dataset.documents
            if ann.document_id == document_id
        ]

    def calculate_mean_human_score(
        self,
        annotations: List[HumanAnnotation]
    ) -> Dict[str, float]:
        """
        Calculate mean scores across multiple annotators.

        Args:
            annotations: List of annotations for same document

        Returns:
            Dict with mean scores for each dimension
        """
        if not annotations:
            return {}

        return {
            "overall_clarity": sum(a.overall_clarity for a in annotations) / len(annotations),
            "instruction_clarity": sum(a.instruction_clarity for a in annotations) / len(annotations),
            "logical_flow": sum(a.logical_flow for a in annotations) / len(annotations),
            "completeness": sum(a.completeness for a in annotations) / len(annotations),
            "consistency": sum(a.consistency for a in annotations) / len(annotations),
            "prerequisite_coverage": sum(a.prerequisite_coverage for a in annotations) / len(annotations),
        }
