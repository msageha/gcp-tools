import logging
from datetime import datetime, timezone
from typing import ClassVar

import validators
from google.auth.credentials import Credentials
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import (
    ContainerSpec,
    CustomJob,
    CustomJobSpec,
    EnvVar,
    MachineSpec,
    Scheduling,
    WorkerPoolSpec,
)
from google.cloud.aiplatform_v1beta1.types.accelerator_type import AcceleratorType
from google.protobuf import duration_pb2
from googleapiclient import errors

from gcp_tools.vertex_ai.utils import (
    DEFAULT_AI_PLATFORM_REGION,
    JobLabels,
    get_project_number,
    region_list,
)


class CustomJobManager:
    # https://cloud.google.com/vertex-ai/docs/training/configure-compute
    machine_type_list: ClassVar[str] = [
        "cloud_tpu",
        "a3-megagpu-8g",
        "a3-highgpu-1g",
        "a3-highgpu-2g",
        "a3-highgpu-4g",
        "a3-highgpu-8g",
        "a2-ultragpu-1g",
        "a2-ultragpu-2g",
        "a2-ultragpu-4g",
        "a2-ultragpu-8g",
        "a2-highgpu-1g",
        "a2-highgpu-2g",
        "a2-highgpu-4g",
        "a2-highgpu-8g",
        "a2-megagpu-16g",
        "e2-standard-4",
        "e2-standard-8",
        "e2-standard-16",
        "e2-standard-32",
        "e2-highmem-2",
        "e2-highmem-4",
        "e2-highmem-8",
        "e2-highmem-16",
        "e2-highcpu-16",
        "e2-highcpu-32",
        "n2-standard-4",
        "n2-standard-8",
        "n2-standard-16",
        "n2-standard-32",
        "n2-standard-48",
        "n2-standard-64",
        "n2-standard-80",
        "n2-highmem-2",
        "n2-highmem-4",
        "n2-highmem-8",
        "n2-highmem-16",
        "n2-highmem-32",
        "n2-highmem-48",
        "n2-highmem-64",
        "n2-highmem-80",
        "n2-highcpu-16",
        "n2-highcpu-32",
        "n2-highcpu-48",
        "n2-highcpu-64",
        "n2-highcpu-80",
        "n1-standard-4",
        "n1-standard-8",
        "n1-standard-16",
        "n1-standard-32",
        "n1-standard-64",
        "n1-standard-96",
        "n1-highmem-2",
        "n1-highmem-4",
        "n1-highmem-8",
        "n1-highmem-16",
        "n1-highmem-32",
        "n1-highmem-64",
        "n1-highmem-96",
        "n1-highcpu-16",
        "n1-highcpu-32",
        "n1-highcpu-64",
        "n1-highcpu-96",
        "c2-standard-4",
        "c2-standard-8",
        "c2-standard-16",
        "c2-standard-30",
        "c2-standard-60",
        "ct5lp-hightpu-1t",
        "ct5lp-hightpu-4t",
        "ct5lp-hightpu-8t",
        "m1-ultramem-40",
        "m1-ultramem-80",
        "m1-ultramem-160",
        "m1-megamem-96",
        "g2-standard-4",
        "g2-standard-8",
        "g2-standard-12",
        "g2-standard-16",
        "g2-standard-24",
        "g2-standard-32",
        "g2-standard-48",
        "g2-standard-96",
    ]

    accelerator_type_list = [i.name for i in AcceleratorType]

    region_list = region_list

    def __init__(
        self,
        project_id: str,
        job_labels: dict[str, str] | JobLabels | None = None,
        region=DEFAULT_AI_PLATFORM_REGION,
        credentials: Credentials | None = None,
    ):
        """
        Args:
            project_id: str.
            job_labels: Dict of str: str.
                Labels to organize jobs.
                The labels with user-defined metadata to organize CustomJobs.
                Label keys and values can be no longer than 64 characters (Unicode codepoints),
                can only contain lowercase letters, numeric characters,
                underscores and dashes. International characters are allowed.
                See https://goo.gl/xmQnxf for more information and examples of labels.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

        """
        self.project_id = project_id
        if isinstance(job_labels, JobLabels):
            job_labels = job_labels.model_dump()
        if job_labels is not None:
            job_labels = {k.lower(): v.lower() for k, v in job_labels.items()}
        self.job_labels = job_labels
        if region not in self.region_list:
            raise ValueError(
                f"Invalid region: {region}",
                f"Please specify in region_list:\n {self.region_list}",
            )
        self.region = region
        self.client = aiplatform.gapic.JobServiceClient(
            client_options={
                "api_endpoint": f"{self.region}-aiplatform.googleapis.com",
            },
            credentials=credentials,
        )

    def deploy_job(
        self,
        image_uri: str,
        entry_point_commands: list[str] | None = None,
        entry_point_args: list[str] | None = None,
        env_vars: list[EnvVar] | None = None,
        machine_type: str = "n1-standard-4",
        accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
        accelerator_count: int = 0,
        worker_count: int = 0,
        worker_machine_type: str | None = None,
        worker_accelerator_type: str | None = None,
        worker_accelerator_count: str | None = None,
        job_id: str | None = None,
        service_account: str | None = None,
        enable_web_access: bool = False,
        network: str | None = None,
        timeout_seconds: int = 60 * 60 * 24,
    ) -> CustomJob:
        """Deploys job with the given parameters to Google Cloud.

        Args:
            image_uri: The Docker image URI.
            entry_point_commands: Optional list of strings. Defaults to None.
                The command to be invoked when the container is started.
                It overrides the entrypoint instruction in Dockerfile when provided.
            entry_point_args: Optional list of strings. Defaults to None.
                The arguments to be passed when starting the container.
            env_vars: Optional list of google.cloud.aiplatform_v1.types.EnvVar. Defaults to None.
                Environment variables to be passed to the container.
                Maximum limit is 100.
            machine_type: str.
            accelerator_type: str.
            accelerator_count: str.
            worker_count: Optional integer that represents the number of general
            workers in a distribution cluster. Defaults to 0. This count does
            not include the chief worker.
            For TPU strategy, `worker_count` should be set to 1.
            worker_machine_type: Optional[str].
            worker_accelerator_type: Optional[str].
            worker_accelerator_count: Optional[str].
            job_id: Optional[str].
                The display name of the CustomJob.
                The name can be up to 128 characters long and can be consist of any UTF-8 characters.
                If it's unspecified, job_id is `platform_integration_train_{"%Y%m%d%H%M"}`.
            service_account: Optional[str].
                The email address of a user-managed service account to be used for training instead of
                the service account that Vertex AI Training uses by default.
                See [custom service account](https://cloud.google.com/ai-platform/training/docs/custom-service-account)
            enable_web_access: bool.
                Whether to enable web access for the job.
                The default value is False.
            network: Optional[str].
                The full name of the Compute Engine network to which the job should be peered.
                For example, `projects/PROJECT_ID/global/networks/NETWORK_NAME`.
                If not specified, the job is not peered with any network.
            timeout_seconds: Optional[int].
                The maximum duration of the job in seconds.
                The job will be automatically cancelled if it runs longer than this duration.
                The default value is 24 hours (86400 seconds).

        Returns:
            google.cloud.aiplatform_v1.types.CustomJob:
                Represents a job that runs custom workloads such as a Docker container or a Python package.
                A CustomJob can have multiple worker pools and each worker pool can have its own machine and
                input spec.
                A CustomJob will be cleaned up once the job enters terminal state (failed or succeeded).

        Raises:
            RuntimeError, if there was an error submitting the job.
        """

        if job_id is None:
            job_id = self._generate_job_id()

        self._validate_machine_config(
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            worker_machine_type=worker_machine_type,
            worker_accelerator_type=worker_accelerator_type,
            worker_accelerator_count=worker_accelerator_count,
            service_account=service_account,
        )

        custom_job = self._create_custom_job(
            display_name=job_id,
            image_uri=image_uri,
            entry_point_commands=entry_point_commands,
            entry_point_args=entry_point_args,
            env_vars=env_vars,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            worker_count=worker_count,
            worker_machine_type=worker_machine_type,
            worker_accelerator_type=worker_accelerator_type,
            worker_accelerator_count=worker_accelerator_count,
            timeout_seconds=timeout_seconds,
            service_account=service_account,
            enable_web_access=enable_web_access,
            network=network,
        )

        try:
            response: CustomJob = self.client.create_custom_job(
                parent=f"projects/{self.project_id}/locations/{self.region}",
                custom_job=custom_job,
            )

            logging.info("Job submitted successfully.")
            logging.info(f"JobID: {job_id}")
            logging.info("Job Information link:")
            _id = response.name.split("/")[-1]
            logging.info(
                f"https://console.cloud.google.com/vertex-ai/locations/{self.region}/training/{_id}/cpu?project={self.project_id}"
            )
            logging.info("Job logs here:")
            logging.info(
                f"https://console.cloud.google.com/logs/query;query=resource.labels.job_id%3D%22{_id}%22?project={self.project_id}",
            )

        except errors.HttpError as err:
            logging.error("There was an error submitting the job.")
            raise err

        return response

    def _validate_machine_config(
        self,
        machine_type,
        accelerator_type,
        accelerator_count,
        worker_machine_type,
        worker_accelerator_type,
        worker_accelerator_count,
        service_account,
    ):
        if machine_type not in self.machine_type_list:
            raise ValueError(
                f"Invalid machine type: {machine_type}",
                f"Please specify in machine_type_list:\n {self.machine_type_list}",
            )
        if (
            accelerator_count == 0
            and accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED"
        ):
            raise ValueError(
                f"Invalid accelerator type: {accelerator_type}",
                "If GPU is not used, do not specify anything for accelerator_type.",
                "If a GPU is used, accelerator_count should be at least 1.",
            )
        if accelerator_count > 0 and accelerator_type not in self.accelerator_type_list:
            raise ValueError(
                f"Invalid accelerator type: {accelerator_type}",
                f"Please specify in accelerator_type_list:\n {self.accelerator_type_list}",
            )
        if (
            worker_machine_type is not None
            and worker_machine_type not in self.machine_type_list
        ):
            raise ValueError(
                f"Invalid worker machine type: {worker_machine_type}",
                f"Please specify in machine_type_list:\n {self.machine_type_list}",
            )
        if worker_accelerator_count == 0 and worker_accelerator_type is not None:
            raise ValueError(
                f"Invalid worker accelerator type: {worker_accelerator_type}",
                "If GPU is not used, do not specify anything for accelerator_type.",
                "If a GPU is used, accelerator_count should be at least 1.",
            )
        if (
            worker_accelerator_type is not None
            and worker_accelerator_type not in self.accelerator_type_list
        ):
            raise ValueError(
                f"Invalid worker accelerator type: {worker_accelerator_type}",
                f"Please specify in accelerator_type_list:\n {self.accelerator_type_list}",
            )
        if service_account is not None and not validators.email(service_account):
            raise ValueError(
                f"Invalid email address of the service account: {service_account}",
            )

    def _create_custom_job(
        self,
        display_name: str,
        image_uri: str,
        entry_point_commands: list[str] | None,
        entry_point_args: list[str] | None,
        env_vars: list[EnvVar] | None,
        machine_type: str,
        accelerator_type: str,
        accelerator_count: int,
        worker_count: int,
        worker_machine_type: str | None,
        worker_accelerator_type: str | None,
        worker_accelerator_count: int | None,
        timeout_seconds: int,
        service_account: str | None = None,
        enable_web_access: bool = False,
        network: str | None = None,
    ):
        worker_pool_specs = []
        if entry_point_args is None:
            entry_point_args = []
        if entry_point_commands is None:
            entry_point_commands = []
        if env_vars is None:
            env_vars = []
        container_spec = ContainerSpec()
        container_spec.image_uri = image_uri
        container_spec.command = entry_point_commands
        container_spec.args = entry_point_args
        container_spec.env = env_vars

        master_machine_spec = MachineSpec()
        master_machine_spec.machine_type = machine_type
        master_machine_spec.accelerator_type = accelerator_type
        master_machine_spec.accelerator_count = accelerator_count

        master_pool_spec = WorkerPoolSpec()
        master_pool_spec.container_spec = container_spec
        master_pool_spec.machine_spec = master_machine_spec
        master_pool_spec.replica_count = 1
        worker_pool_specs.append(master_pool_spec)

        if worker_count > 0:
            worker_machine_spec = MachineSpec()
            worker_machine_spec.machine_type = worker_machine_type
            worker_machine_spec.accelerator_type = worker_accelerator_type
            worker_machine_spec.accelerator_count = worker_accelerator_count

            worker_pool_spec = WorkerPoolSpec()
            worker_pool_spec.container_spec = container_spec
            worker_pool_spec.machine_spec = worker_machine_spec
            worker_pool_spec.replica_count = worker_count
            worker_pool_specs.append(worker_pool_spec)

        scheduling = Scheduling()
        scheduling.timeout = duration_pb2.Duration(seconds=timeout_seconds)
        scheduling.disable_retries = True

        custom_job_spec = CustomJobSpec()
        custom_job_spec.worker_pool_specs = worker_pool_specs
        custom_job_spec.scheduling = scheduling
        if service_account is not None:
            custom_job_spec.service_account = service_account
        custom_job_spec.enable_web_access = enable_web_access

        if network:
            project_number = get_project_number(self.project_id)
            custom_job_spec.network = (
                f"projects/{project_number}/global/networks/{network}"
            )

        request = CustomJob()
        request.display_name = display_name
        request.job_spec = custom_job_spec
        request.labels = self.job_labels

        return request

    @staticmethod
    def _generate_job_id():
        """Returns a unique job id prefixed with 'platform_integration_train'."""
        utc_time = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M")
        return f"platform_integration_train_{utc_time}"

    def list_jobs(self) -> list[CustomJob]:
        jobs = self.client.list_custom_jobs(
            parent=f"projects/{self.project_id}/locations/{self.region}",
        )
        jobs = list(jobs)
        return jobs

    def get_job(
        self,
        name: str | None = None,
        custom_job: int | None = None,
    ) -> CustomJob:
        if custom_job is None and name is None:
            raise ValueError
        elif custom_job is not None:
            name = f"projects/{self.project_id}/locations/{self.region}/customJobs/{custom_job}"

        job = self.client.get_custom_job(name=name)
        return job

    def get_job_url(
        self,
        name: str | None = None,
        custom_job: int | None = None,
    ) -> str:
        if custom_job is None and name is None:
            raise ValueError
        elif custom_job is not None:
            name = f"projects/{self.project_id}/locations/{self.region}/customJobs/{custom_job}"

        _id = name.split("/")[-1]
        url = f"https://console.cloud.google.com/vertex-ai/locations/{self.region}/training/{_id}/cpu?project={self.project_id}"
        return url

    def cancel_job(
        self,
        name: str | None = None,
        custom_job: int | None = None,
    ):
        if custom_job is None and name is None:
            raise ValueError
        elif custom_job is not None:
            name = f"projects/{self.project_id}/locations/{self.region}/customJobs/{custom_job}"

        self.client.cancel_custom_job(name=name)


if __name__ == "__main__":
    import google.auth

    credentials, project_id = google.auth.default()
    if project_id is None:
        raise ValueError("project_id is not set in the environment.")
    image_uri = "hello-world:latest"

    job_labels = JobLabels(
        task="test",
    )

    manager = CustomJobManager(
        project_id,
        job_labels,
    )

    job_info = manager.deploy_job(image_uri, machine_type="n1-standard-4")
    url = manager.get_job_url(name=job_info.name)
    print(url)
