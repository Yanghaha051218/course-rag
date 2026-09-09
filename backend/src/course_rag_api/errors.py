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
