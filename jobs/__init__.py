import gin
from jobs.job import Job
from jobs.full_pipeline import FullPipeline


def configure_class(class_object):
    return gin.external_configurable(class_object, module='jobs')


Job = configure_class(Job)
FullPipeline = configure_class(FullPipeline)
