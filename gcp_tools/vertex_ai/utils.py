from datetime import UTC, datetime
from enum import StrEnum

from google.cloud import resourcemanager_v3
from pydantic import BaseModel, Field

DEFAULT_AI_PLATFORM_REGION = "us-central1"


class OrderByType(StrEnum):
    DISPLAY_NAME = "display_name"
    DISPLAY_NAME_DESC = "display_name desc"
    NAME = "name"
    NAME_DESC = "name desc"
    CREATE_TIME = "create_time"
    CREATE_TIME_DESC = "create_time desc"
    UPDATE_TIME = "update_time"
    UPDATE_TIME_DESC = "update_time desc"


class JobLabels(BaseModel):
    task: str = Field(..., description="The task name for the job.")
    batch_id: str = Field(
        default=datetime.now(UTC).strftime("%Y%m%d%H%M"),
        description="The batch ID for the job.",
    )

    def get_job_id(self):
        job_id = f"{self.task}_{self.batch_id}"
        return job_id


def get_project_number(project_id: str) -> str:
    """
    Get the project number for a given project ID.
    """
    client = resourcemanager_v3.ProjectsClient()
    project = client.get_project(name=f"projects/{project_id}")
    return project.name.split("/")[1]


region_list = [
    "us-central1",
    "us-east1",
    "us-east4",
    "us-west1",
    "us-west2",
    "northamerica-northeast1",
    "northamerica-northeast2",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast3",
    "asia-south1",
    "asia-southeast1",
    "australia-southeast1",
]
