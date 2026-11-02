"""TinyAudio: compact and efficient text-to-audio generation."""

__version__ = '1.0.0'
__all__ = ['TinyAudioPipeline', 'download_checkpoints']


def __getattr__(name):
    if name in __all__:
        from tinyaudio import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
