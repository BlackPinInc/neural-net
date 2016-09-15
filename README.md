# Build

cabal sandbox init
cabal install c2hs
cabal install --only-dependencies

# Info

The project is divided into a few folders:
  
 - src/ -- The directory that contains the library's source
 - app/ -- Contains the neural-net executable
 - bench/ -- Benchmarking executable
 - test/ -- unit tests

## src/ 

This folder contains the NeuralNet library. Haskell has a flat module inclusion
pattern which means that all modules must have a descriptive prefix. This 
module works only with and uses the accelerate library so we are using the
accelerate modules prefix scheme Data.Array.Accelerate. NeuralNet is at 
Data.Array.Accelerate.NeuralNet. This is so that it can be sent to hackage and
used without confusion. I want this one to be used by as many people as possible.
Although this module structure is nice for hackage it sucks when you are trying
to edit the code because you have to type 
`vim Data/Array/Accelerate/NeuralNet/Layer/LayerCombiners.hs` each time you
edit a file. Thats why I made symlinks to each file that arent included in the
library. The modules in this library are:

 - Data.Array.Accelerate.NeuralNet

  The only visible module. This module imports all the other modules

 - Data.Array.Accelerate.NeuralNet.Layer
  
  This module imports all the contained modules.

  + Data.Array.Accelerate.NeuralNet.Layer.Types

    This module contains the type defititions for a `Layer`.  The reason I put
    this in its own module is to avoid circular imports. In general tree
    structures help avoid circular imports.

  + Data.Array.Accelerate.NeuralNet.Layer.DynamicLayers

    This module contains functions which create dynamic layers. These are layers 
    which have an internal `weight` parameter which changes when SGD is applied
    to it. This includes `MatMulLayer`, `BiasLayer`, and would include `ConvLayer`s.

  + Data.Array.Accelerate.NeuralNet.Layer.StaticLayers

    This module contains functions which create static layers. Static layers are
    layers without a `weight` parameter. These layers are not changed by gradient
    descent. This includes `ActivationLayer`, `NormalizeLayer`, `ReshapeLayer`, 
    and `FlattenLayer`.

  + Data.Array.Accelerate.NeuralNet.Layer.LayerCombiners

    This module contains functions which combine `Layer`s to create more
    complicated layers. The most important and descriptive is the `TupleLayer`,
    which creates a layer with two layers and it combines their functions and 
    weights with tuples.  Is there a Monad in here? IDK.

 - Data.Array.Accelerate.NeuralNet.Loader

  Loads data into the NeuralNet environment. At the moment its only mnist.

 - Data.Array.Accelerate.NeuralNet.Activation

  Provides a definition for an activation function which can be implemented.

 - Data.Array.Accelerate.NeuralNet.CostFunction

  Provides a definition for a cost function which can also be implemented.

 - Data.Array.Accelerate.NeuralNet.Util

  Holds utility functions.
   


# TODO:

## Jordan
Finish this god damn thing.
So whats left?
Speed, Usability

Speed is really the biggest thing.
I need problem solving help most.

