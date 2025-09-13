# GCP Tools

## How to install

```bash
pip install --upgrade git+https://github.com/msageha/gcp-tools.git
```

## How to use

### Vertex AI

- custom training job
```python
from gcp_tools.vertex_ai.utils import (
    JobLabels,
)
from gcp_tools.vertex_ai.custom_job import CustomJobManager

job_labels = JobLabels(
    task="test",
)
manager = CustomJobManager(
    project_id="your-project-id",
    job_labels=job_labels,
)
job_info = manager.deploy_job(
    image_uri="hello-world:latest",
    machine_type="n1-standard-4",
)
print(manager.get_job_url(name=job_info.name))
```
