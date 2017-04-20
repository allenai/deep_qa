"""
The Trainer code needs to be able to create Pretrainer objects just from a Dict object that's
passed  to it.  In order to make that work, we keep a dictionary here that maps names to classes,
so we can create a Pretrainer using a string parameter.

However, in this module, we have no idea what pretrainers you've implemented, so we can't just
import them all.  If we did that, anyone who wanted to use this code as a library would have to
modify this file, adding their own pretrainer.  Instead, you should _modify_ this object in the
__init__.py module of your code, adding whatever pretrainers you need to use.  That way we can
create your pretrainer objects without having to know about them in this module.
"""
from .text_trainer import TextTrainer
from .trainer import Trainer

concrete_pretrainers = {  # pylint: disable=invalid-name
        }
