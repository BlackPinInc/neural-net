{-# LANGUAGE TypeOperators #-}
module NetSpec (main, spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Array.Accelerate.NeuralNet
import Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.CUDA as I
import Control.Exception (evaluate)


-- `main` is here so that this module can be run from GHCi on its own.  It is
-- not needed for automatic spec discovery.
main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Sigmoid" $ do
    let neginf = (-1/0) -- -inf
    let posinf = (1/0) -- +inf
    let zero = 0 -- 0
    let testSingleton f x = indexArray (I.run (f (unit x))) Z

    it "is 0 at -inf" $ do
      testSingleton (apply Sigmoid) neginf `shouldBe` 0
    it "is 1 at inf" $ do
      testSingleton (apply Sigmoid) posinf `shouldBe` 1
    it "is 0.5 at 0" $ do
      testSingleton (apply Sigmoid) zero `shouldBe` 0.5

    it "has a delta of 0 at -inf" $ do
      testSingleton (delta Sigmoid) neginf `shouldBe` 0
    it "has a delta of 0 at inf" $ do
      testSingleton (delta Sigmoid) posinf `shouldBe` 0
    it "has a delta of 0.25 at 0" $ do
      testSingleton (delta Sigmoid) zero `shouldBe` 0.25

  describe "Matrix Multiplication Layer (MatMulLayer, mkMatMulLayer)" $ do
    let layer = mkUnitMatMulLayer 1 10 5
    let input = use $ fromList (Z:.10) [1..]
    it "takes a vector of one length and returns another" $ do
      let outputVec = I.run $ feedForward layer (param layer) input
      let (Z:.x) = arrayShape outputVec 
      x `shouldBe` 5
    -- it "should throw an error on the wrong input size" $ do
      -- evaluate (run $ layer `feedForward` input2) `shouldThrow` anyException
  describe "Bias Layer (BiasLayer, mkBiasLayer)" $ do
    let layer = mkUnitBiasLayer 1 5
    let input = use $ fromList (Z:.5) [1..]
    it "takes a vector and returns the elementwize sum with an internal mat" $ do
      let accVec = feedForward layer (param layer) input
      let boolValue = A.and $ A.zipWith (==*) accVec (enumFromN (index1 5) 2)
      let finalBool = I.run boolValue
      indexArray finalBool Z `shouldBe` True
  describe "Activation Layer (ActivationLayer mkActivationLayer)" $ do
    let layer = mkSigmoidLayer 5
    let input = use $ fromList (Z:.5) $ repeat (-1/0)
    it "takes a vector of -inf and returns the elementwise application of sigmoid aka ret=0" $ do
      let accVec = feedForward layer (param layer) input
      let boolValue = A.and $ A.zipWith (==*) accVec (use $ fromList (Z:.5) $ repeat 0)
      let finalBool = indexArray (I.run boolValue) Z
      finalBool `shouldBe` True
  describe "Network Layer" $ do
    it "should combine two layers" $ do
      let layer = mkNetwork (mkUnitMatMulLayer 1 10 5) (mkUnitBiasLayer 1 5)
      let input = enumFromN (index1 10) 1
      let accVec = feedForward layer (param layer) input
      let expected = use $ fromList (Z:.5) $ repeat 56 -- sum [1..10] + 1
      let boolValue = A.and $ A.zipWith (==*) accVec expected
      let finalBool = indexArray (I.run boolValue) Z
      finalBool `shouldBe` True
