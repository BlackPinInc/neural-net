module Data.Array.Accelerate.NeuralNet.Layer.DynamicLayers (
  MatMulLayer, mkMatMulLayer, mkUnitMatMulLayer, mkNormalMatMulLayer,
  BiasLayer, mkBiasLayer, mkUnitBiasLayer, mkNormalBiasLayer
) where

import Prelude as P
import Data.Array.Accelerate 
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Arithmetic.LinearAlgebra as T

import Data.Array.Accelerate.NeuralNet.Layer.Types
import Data.Array.Accelerate.NeuralNet.Util


-- Default functions
lc :: (Elt e, IsNum e, Shape sh) => Acc (Array sh e) -> Exp e
lc = A.the . A.sum
re :: (Elt e, IsNum e, Shape sh) => Acc (Array sh e) -> Acc (Array sh e) -> Acc (Array sh e)
re = A.zipWith (-)
ce :: (Elt e, IsNum e, Shape sh) => Acc (Array sh e) -> Acc (Array sh e) -> Acc (Array sh e)
ce = A.zipWith (+)


-- MatMulLayer --
                               
-- | Matrix multiplication layer. Takes an input vector and returns the weight
-- matrix multiplied by that vector.
type MatMulLayer = Layer (T.Matrix Z Float) DIM1 DIM1 Float Float DIM1 DIM1

-- | Create a new matrix multiplication layer.
mkMatMulLayer :: T.Matrix Z Float -- ^ The initial weight matrix.
              -> Int              -- ^ The length of the input vector.
              -> Int              -- ^ The length of the output vector.
              -> MatMulLayer      -- ^ The output Layer
mkMatMulLayer init inSize outSize = layer
  where weights = init -- TODO: randoms instread of 1
        ff w input = T.multiplyMatrixVector w input
        fb w input prevDeriv = (dw, T.vectorFromColumn d)
          where dw = T.multiplyMatrixMatrix d (T.transpose $ T.columnFromVector input)
                d = (T.transpose w) `T.multiplyMatrixMatrix` (T.columnFromVector prevDeriv)
        layer :: MatMulLayer
        layer = Layer { inputSize = index1 $ lift inSize
                      , outputSize = index1 $ lift outSize
                      , param = weights
                      , feedForward = ff
                      , feedBack = fb
                      , layerCost = lc
                      , removeError = re
                      , combineError = ce
                      }

-- | Create a new matrix multiplication layer with the matrix initialized with
-- a single element value.
mkUnitMatMulLayer :: Float        -- ^ The initial value of the layer.
                  -> Int          -- ^ The length of the input vector.
                  -> Int          -- ^ The length of the output vector.
                  -> MatMulLayer  -- ^ The output Layer.
mkUnitMatMulLayer i inSize outSize = mkMatMulLayer init inSize outSize
  where init = fill (index2 (lift outSize) (lift inSize)) $ lift i

-- | Create a new matrix multiplication layer initialized with normally 
-- distributed random numbers.
mkNormalMatMulLayer :: Float          -- ^ Mean for the RNG.
                    -> Float          -- ^ Standard deviation for the RNG.
                    -> Int            -- ^ The length of the input vector.
                    -> Int            -- ^ The length of the output vector.
                    -> IO MatMulLayer -- ^ The output Layer.
mkNormalMatMulLayer m s inSize outSize = do
  init <- normalArray m s (Z:.outSize:.inSize)
  return $ mkMatMulLayer init inSize outSize


-- BiasLayer -- 

-- | Bias layer adds the input to a bias vector.
type BiasLayer = Layer (T.Vector Z Float) DIM1 DIM1 Float Float DIM1 DIM1

-- | Create a bias layer.
mkBiasLayer :: T.Vector Z Float -- ^ Initial bias vector.
            -> Int              -- ^ The size of the input and output vector.
            -> BiasLayer        -- ^ The output Layer.
mkBiasLayer init size = layer
  where biases = init -- TODO: randoms?
        ff b input = A.zipWith (+) input b
        fb _ _ p = (p, p)
        layer = Layer { inputSize = index1 $ lift size
                      , outputSize = index1 $ lift size
                      , param = biases
                      , feedForward = ff
                      , feedBack = fb
                      , removeError = re
                      , combineError = ce
                      , layerCost = lc
                      }

-- | Create a bias layer using a single value to initalize the vector.
mkUnitBiasLayer :: Float      -- ^ The value to initialize the vector with.
                -> Int        -- ^ The input and output vector length.
                -> BiasLayer  -- ^ The output Layer.
mkUnitBiasLayer i size = mkBiasLayer (fill (index1 $ lift size) $ lift i) size 

-- | Create a bias layer with a normally distributed random vector.
mkNormalBiasLayer :: Float        -- ^ Mean for the RNG.
                  -> Float        -- ^ Standard deviation for the RNG.
                  -> Int          -- ^ Input/Output vector length.
                  -> IO BiasLayer -- ^ The output Layer.
mkNormalBiasLayer m s size = do
  init <- normalArray m s (Z:.size)
  return $ mkBiasLayer init size

 
