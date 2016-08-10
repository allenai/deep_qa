This code handles tree transformations and logical form generation.

We have an abstract interface to an actual dependency parser, and so theoretically can work with
any parser you want (though currently we only have a concrete implementation for the Stanford
parser).  We just need a particular representation of a dependency tree, which we then transform
into a logical form, or allow other kinds of transformations as desired.
