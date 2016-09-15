{-# LANGUAGE ExistentialQuantification, ScopedTypeVariables, MultiParamTypeClasses, TypeSynonymInstances, FlexibleInstances, TypeFamilies #-}
module Data.Array.Accelerate.NeuralNet.Layer.LayerCombiners (
  pushStaticLayerFront,
  TupleLayer, mkTupleLayer,
  FullyConnectedLayer, mkFullyConnectedLayer,
  Network(..), netFeedForward{-, netSGD,
  (<!>)-}
) where

import Prelude as P
import Data.List as P
import System.Random (randoms, newStdGen)
import Control.Applicative ((<$>))
import GHC.Exts (sortWith)
import Data.Foldable (foldlM)
import Control.Monad (forM)
import Control.Monad.Trans (liftIO)
import Control.Monad.Trans.State (StateT(..), get, put)

import Data.Array.Accelerate 
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Arithmetic.LinearAlgebra as T

import Data.Array.Accelerate.NeuralNet.Activation
import Data.Array.Accelerate.NeuralNet.CostFunction
import Data.Array.Accelerate.NeuralNet.Layer.Types
import Data.Array.Accelerate.NeuralNet.Layer.StaticLayers
import Data.Array.Accelerate.NeuralNet.Layer.DynamicLayers

                
pushStaticLayerBack :: Layer w d oDeriv iType t iSize s
                    -> StaticLayer iDeriv d t oType s oSize 
                    -> Layer w iDeriv oDeriv iType oType iSize oSize
pushStaticLayerBack l s = layer
  where ff w input = feedForward s () $ feedForward l w input
        fb w input prevDeriv = (nw, out1)
          where input2 = feedForward l w input
                ((), out2) = feedBack s () input2 prevDeriv
                (nw, out1) = feedBack l w input out2
        layer = l { inputSize = (inputSize l)
                  , outputSize = (outputSize s)
                  , feedForward = ff
                  , feedBack = fb
                  }

pushStaticLayerFront :: StaticLayer d oDeriv iType t iSize s
                     -> Layer w  iDeriv d t oType s oSize 
                     -> Layer w iDeriv oDeriv iType oType iSize oSize
pushStaticLayerFront s l = layer
  where ff w input = feedForward l w $ feedForward s () input
        fb w input prevDeriv = (nw, out1)
          where input2 = feedForward s () input
                (nw, out2) = feedBack l w input2 prevDeriv
                ((), out1) = feedBack s () input out2
        layer = l { inputSize = (inputSize s)
                  , outputSize = (outputSize l)
                  , feedForward = ff
                  , feedBack = fb
                  }
-- | Tuple Layer

-- | Tuple Layer to combine two layers.
type TupleLayer w1 w2 iDeriv oDeriv iType oType iSize oSize 
     = Layer (w1, w2) iDeriv oDeriv iType oType iSize oSize

mkTupleLayer :: Layer w1 d oDeriv1 iType1 t iSize1 s -- ^ Layer #1
             -> Layer w2 iDeriv2 d t oType2 s oSize2 -- ^ Layer #2
             -> TupleLayer w1 w2 iDeriv2 oDeriv1 iType1 oType2 iSize1 oSize2 
mkTupleLayer layer1 layer2 = net
  where p = (param layer1, param layer2)
        ff (w1, w2) input = feedForward layer2 w2 out
          where out = feedForward layer1 w1 input
        fb (w1, w2) input prevDeriv = ((nw1, nw2), out1)
          where input2 = feedForward layer1 w1 input
                (nw2, out2) = feedBack layer2 w2 input2 prevDeriv
                (nw1, out1) = feedBack layer1 w1 input out2
        re (w1, w2) (nw1, nw2) = ( removeError layer1 w1 nw1
                                 , removeError layer2 w2 nw2
                                 )
        ce (w1, w2) (nw1, nw2) = ( combineError layer1 w1 nw1
                                 , combineError layer2 w2 nw2
                                 )
        lc (w1, w2) = layerCost layer1 w1 + layerCost layer2 w2
        net = Layer { param = p
                    , inputSize = (inputSize layer1)
                    , outputSize = (outputSize layer2)
                    , feedForward = ff
                    , feedBack = fb
                    , removeError = re
                    , combineError = ce
                    , layerCost = lc
                    }
                
-- | Fully connected layer.  Classic fully connected neural network layer.
type FullyConnectedLayer 
  = Layer (T.Matrix Z Float, T.Vector Z Float) DIM1 DIM1 Float Float DIM1 DIM1

-- | Create a fully connected layer.
mkFullyConnectedLayer :: (Activation a) 
                      => a                      -- ^ The activation function.
                      -> Int                    -- ^ The input vector length.
                      -> Int                    -- ^ The output vector length.
                      -> IO FullyConnectedLayer -- ^ The output Layer
mkFullyConnectedLayer a i o = do
  matMulLayer <- mkNormalMatMulLayer 0 (sqrt (P.fromIntegral i)) i o
  biasLayer <- mkNormalBiasLayer 0 1 o
  let actLayer = mkActivationLayer a (index1 $ constant o)
  let p = (param matMulLayer, param biasLayer)
  let ff (w, b) input = let wx = feedForward matMulLayer w input
                            z = feedForward biasLayer b wx
                            a = feedForward actLayer () z
                        in a
  let fb (w, b) input prevDeriv = let wx = feedForward matMulLayer w input
                                      z = feedForward biasLayer b wx
                                      (_, d1) = feedBack actLayer () z prevDeriv
                                      (nb, d2) = feedBack biasLayer b wx d1
                                      (nw, newDeriv) = feedBack matMulLayer w input d2
                                  in ((nw, nb), newDeriv)
  let re (w, b) (nw, nb) = ( removeError matMulLayer w nw
                           , removeError biasLayer b nb
                           )
  let ce (w, b) (nw, nb) = ( combineError matMulLayer w nw
                           , combineError biasLayer b nb
                           )
  let lc (w, b) = layerCost matMulLayer w + layerCost biasLayer b
  let net = Layer { param = p
                  , inputSize = index1 $ lift i
                  , outputSize = index1 $ lift o
                  , feedForward = ff
                  , feedBack = fb
                  , removeError = re
                  , combineError = ce
                  , layerCost = lc
                  }
  return net


-- | Multi layer network (aka list of layers)
type MultiLayerNetwork w d t s = Layer [w] d d t t s s

-- | Creates a null (identity) layer.
nullNetwork :: MultiLayerNetwork w d t s
nullNetwork = Layer { param = []
                    , feedForward = const id
                    , feedBack = \_ _ p -> ([],p)
                    , removeError = const $ const $ []
                    , combineError = const $ const $ []
                    , layerCost = const 0
                    }

-- | Creates a new Layer by adding a layer to a network.
consNetwork :: Layer w d d t t s s        -- ^ The Layer to add to the network.
            -> MultiLayerNetwork w d t s  -- ^ The Network to be added to.
            -> MultiLayerNetwork w d t s  -- ^ The output Network.
consNetwork layer net = net
  where ff (w:ws) input = feedForward net ws $ feedForward layer w input
        -- This looks crazy like theres no base case for the recursion but there isnt
        -- because this isnt recursive.
        -- it calls `feedForward` in `net` not in the current network!
        -- This code is blowing my fucking mind.
        fb (w:ws) input prevDeriv = (nw:nws, d2)
          where act = feedForward layer w input 
                (nws, d1) = feedBack net ws act prevDeriv
                (nw, d2) = feedBack layer w input d1
        -- This code runs `feedForward` all the way down the chain then comes back up with 
        -- the backpropagation. This is because the recursive step is between the individual steps.
        re (w:ws) (nw:nws) = (removeError layer w nw : removeError net ws nws)
        ce (w:ws) (nw:nws) = (combineError layer w nw : combineError net ws nws)
        lc (w:ws) = layerCost layer w + layerCost net ws
        net = Layer { param = (param layer) : (param net)
                    , inputSize = (inputSize layer) 
                    , outputSize = if (P.null $ param net) then (outputSize layer) else (outputSize net)
                    , feedForward = ff
                    , feedBack = fb
                    , removeError = re
                    , combineError = ce
                    , layerCost = lc
                    }

-- | Take a list of layers and turn it into a Network.
mkMultiLayerNetwork :: [Layer w d d t t s s] -> MultiLayerNetwork w d t s 
mkMultiLayerNetwork [] = nullNetwork
mkMultiLayerNetwork (x:xs) = x `consNetwork` mkMultiLayerNetwork xs

                
-- | Network type to take a layer and a cost function and use them to train a net.
data Network w s i o = forall c . CostFunction c => 
                 Network { innerLayer :: Layer w DIM1 DIM1 i o s DIM1
                         , costFunction :: c
                         }

-- | Feed forward through a network.
netFeedForward :: Network w s i o -> Acc (Array s i) -> Acc (Array DIM1 o)
netFeedForward net = feedForward layer (param layer) 
  where layer = innerLayer net

{-
-- | Feed backward though a network.
netBackprop :: (Elt i) => Network w s i Float -> Acc (Array s i, Scalar Int) -> w
netBackprop (Network layer cf) arrn = nab
  where (arr, n) = unlift arrn
        output = feedForward layer (param $ layer) arr
        prevDeriv = cf `cost_delta` output $ the n
        (nab, deriv) = feedBack layer (param $ layer) arr prevDeriv
 
-- | Update a network using a batch of input output pairs.
netUpdateMiniBatch :: (Elt i, Shape s) => Network w s i Float -> [(Array s i, Scalar Int)] -> Network w s i Float
netUpdateMiniBatch net batch = newNet
  where newWeight = removeError (innerLayer net) (param $ innerLayer net) $ netBackprop net $ unlift $ use $ batch
        newNet = net { innerLayer = (innerLayer net) { param = newWeight } }

shuffle :: [a] -> IO [a]
shuffle xs = do
  rs <- randoms <$> newStdGen :: IO [Int]
  return $ P.map P.fst $ sortWith P.snd $ P.zip xs rs

batched :: Int -> [a] -> [[a]]
batched n xs = if P.null b then [] else a : batched n b
  where (a, b) = splitAt n xs

stateForM state xs f = foldlM f state xs

netSGD :: Network w s i Float
       -> [(Array s i, Scalar Int)]
       -> [(Array s i, Scalar Int)]
       -> Int
       -> Int 
       -> Double
       -> Double
       -> IO (Network w s i Float)
netSGD net trainingData evaluationData epochs batchSize eta lambda = do
  net <- stateForM net [1..epochs] $ \net epoch -> do 
    shuffled <- liftIO $ shuffle trainingData
    let batches = batched batchSize shuffled
    foldlM netUpdateMiniBatch net batches
  return net
  

netCost :: Network w s i Float -> [(Array s i, Scalar Int)] -> Scalar Int
netCost net@(Network l c) sets = summed
  where activations = netFeedForward net $ P.map P.fst sets
        costs = P.zipWith (cost_apply c) activations $ P.map P.snd sets
        summed = (layerCost l (param l)) + (P.sum costs)

netAccuracy :: Network w s i Float -> [(Array s i, Scalar Int)] -> Scalar Int
netAccuracy net sets = P.sum $ P.map boolToInt $ P.zipWith (==*) activations $ P.map P.snd sets
  where activations = P.map (unindex1.argmax) $ netFeedForward net $ P.map P.fst sets


infixr 5 <!>

(<!>) = mkTupleLayer
-}
