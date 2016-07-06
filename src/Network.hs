{-# LANGUAGE FlexibleContexts, TupleSections #-}
module Network where

import Data.Matrix
import Data.Vector (singleton)
import Data.Maybe (isJust, fromJust)
import Data.Random.Normal
import System.Random

data Network = Network { numLayers :: Int
                       , sizes :: [Int]
                       , biases :: [Matrix Double]
                       , weights :: [Matrix Double]
                       } deriving (Show)

instance Fractional a => Fractional (Matrix a) where
  (/) = elementwise (/)
  fromRational = colVector . singleton . fromRational

instance Floating a => Floating (Matrix a) where
  pi = colVector $ singleton pi
  exp = fmap exp
  log = fmap log
  sin = fmap sin
  cos = fmap cos
  asin = fmap asin
  acos = fmap acos
  atan = fmap atan
  sinh = fmap sinh
  cosh = fmap cosh
  asinh = fmap asinh
  acosh = fmap acosh
  atanh = fmap atanh


biasSizes = map (1,) . tail

weigthSizes sizes = zip sizes $ tail sizes

randn g (w,h) = fromList h w $ normals g

tonORandoms g = g1 : tonORandoms g2
  where (g1, g2) = split g

mkNetwork :: [Int] -> IO Network
mkNetwork widths = do
  stdGen <- getStdGen
  stdGen1 <- getStdGen
  return Network { numLayers = length widths
                 , sizes = widths
                 , biases = zipWith randn (tonORandoms stdGen) $ biasSizes widths
                 , weights = zipWith randn (tonORandoms stdGen1) $ weigthSizes widths}

-- Thing about what haskell does best.
-- is it algorithms or equations?
-- Avoid concrete types, use Num, Floating, or Functor instead of Matrix.
-- If you need to make your own class and have Matrix implement that class maybe you should.
-- I looked at the equations again last night and they will probably be easier to implement
-- in Haskell than the algorithms.
--
-- The feed forward algorithm has an equation that we can implement directly in Haskell.
-- No need to follow their code.  Or my own crappy code.  Blind leading the blind.
--
-- Take a look at the backprop equations(BP1-BP4) I can help you translate these directly
-- into haskell functions.  All of these equations have a direct haskell equivalent.
--
-- And you should look at this link for how to set up the unit testing.
-- https://github.com/kazu-yamamoto/unit-test-example
-- They use HSpec instead of HUnit but I would recomend HSpec because its more haskelly.
--
-- Lol sorry for all the text I am just very excited about this project and I want you to
-- have a better time on this than I did.

sigmoid :: (Floating (f b), Fractional b, Functor f) => f b -> f b
sigmoid z = fmap (1/) $ fmap (1+) (exp $ negate z)

sigmoid' :: Floating c => Matrix c -> Matrix c
sigmoid' z = elementwise (*) (sigmoid z) (1-sigmoid(z))

feedForward :: Network -> Matrix Double -> Matrix Double
feedForward net x
  | (numLayers net) >= 1 = feedForward newNet a
  | otherwise            = a
    where a = sigmoid (elementwise (*) (head $ weights net) x) + (head $ biases net)
          newNet = Network { numLayers = numLayers net - 1
                           , sizes = tail (sizes net)
                           , biases = tail (biases net)
                           , weights = tail (weights net)}

-- cost = 1/2*n*sum(magnitude(output-activation)^2)
-- C(w,b)≡1/(2n)∑x ∥ y(x)−a ∥^2.
