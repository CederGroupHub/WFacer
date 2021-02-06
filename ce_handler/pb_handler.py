from abc import ABC,abstractmethod

from .base import BaseHandler

class PBHandler(BaseHandler,ABC):
     def __init__(self):
         pass

class CanonicalPBHandler(PBHandler):
     def __init__(self):
         pass

class SemigrandPBHandler(PBHandler):
     def __init__(self):
         pass
