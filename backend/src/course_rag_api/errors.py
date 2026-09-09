class CourseRAGError(Exception):
    """Base class for expected ingestion failures."""


class MissingDocumentError(CourseRAGError):
    pass


class UnsupportedFileTypeError(CourseRAGError):
    pass


class EmptyDocumentError(CourseRAGError):
    pass


class CourseNotFoundError(CourseRAGError):
    pass


class DocumentTooLargeError(CourseRAGError):
    pass


class ConfigurationError(CourseRAGError):
    pass


class EmbeddingError(CourseRAGError):
    pass


class DocumentNotFoundError(CourseRAGError):
    pass


class IndexCompatibilityError(CourseRAGError):
    pass


class IndexConsistencyError(CourseRAGError):
    pass
