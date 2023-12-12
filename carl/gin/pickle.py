from abc import ABC, abstractmethod
import copy
import inspect


class CarlPickle:

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def unpack(self):
        return self.cls(**{k: v.unpack() if isinstance(v, CarlPickle) else v for k, v in self.kwargs.items()})


class CarlPickable(ABC):

    def inspect_class(self):
        raise NotImplementedError

    def carl_pickle(self) -> CarlPickle:

        args = inspect.getfullargspec(self.inspect_class().__init__).args[1:]

        arg_dict = {
            a: copy.copy(getattr(self, a))
            if not isinstance(getattr(self, a), CarlPickable) else getattr(self, a).carl_pickle() for a in args
        }

        pickle_obj = CarlPickle(self.inspect_class(), **arg_dict)
        return pickle_obj


class CarlConstructable(ABC):
    """
    This is a class that signalizes that Constructable instances are constructed only from the
    attributes which are pickable. If something in normal environment is not pickable, it should be Constructable too with similar manner. If something is not our code but still pickable, then it should be constrructed in overrided carl_construct method.
    """

    def carl_construct(self) -> None:
        """Takes all the arguments from the constructor and then builds proper objects"""
        # Iterate over all the attributes and if they are constructable, call construct on them
        for attr, value in self.__dict__.items():
            if isinstance(value, CarlConstructable):
                value.carl_construct()
