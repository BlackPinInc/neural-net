{-# LANGUAGE TypeOperators #-}
module NetSpec (main, spec) where

import Test.Hspec
import Test.QuickCheck
import Control.Arrow ((***))
import Data.Array.Accelerate.NeuralNet
import Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Interpreter as I
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
      testSingleton (act_apply Sigmoid) neginf `shouldBe` 0
    it "is 1 at inf" $ do
      testSingleton (act_apply Sigmoid) posinf `shouldBe` 1
    it "is 0.5 at 0" $ do
      testSingleton (act_apply Sigmoid) zero `shouldBe` 0.5

    it "has a delta of 0 at -inf" $ do
      testSingleton (act_delta Sigmoid) neginf `shouldBe` 0
    it "has a delta of 0 at inf" $ do
      testSingleton (act_delta Sigmoid) posinf `shouldBe` 0
    it "has a delta of 0.25 at 0" $ do
      testSingleton (act_delta Sigmoid) zero `shouldBe` 0.25

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
    let layer = mkSigmoidLayer (index1 $ constant 5) :: ActivationLayer DIM1
    let input = use $ fromList (Z:.5) $ repeat (-1/0)
    it "takes a vector of -inf and returns the elementwise application of sigmoid aka ret=0" $ do
      let accVec = feedForward layer (param layer) input
      let boolValue = A.and $ A.zipWith (==*) accVec (use $ fromList (Z:.5) $ repeat 0)
      let finalBool = indexArray (I.run boolValue) Z
      finalBool `shouldBe` True
  describe "Tuple Layer" $ do
    it "should combine two layers" $ do
      let layer = mkUnitMatMulLayer 1 10 5 <!> mkUnitBiasLayer 1 5
      let input = enumFromN (index1 10) 1
      let accVec = feedForward layer (param layer) input
      let expected = use $ fromList (Z:.5) $ repeat 56 -- sum [1..10] + 1
      let boolValue = A.and $ A.zipWith (==*) accVec expected
      let finalBool = indexArray (I.run boolValue) Z
      finalBool `shouldBe` True
  let isInRange l h = (flip indexArray Z) . I.run . A.all (\x -> l <=* x &&* x <=* h)
  let allInRange l h = Prelude.all (isInRange l h)
  describe "Fully Connected Layer" $ do
    it "always returns a vector between 0 and 1" $ do
      layer <- mkFullyConnectedLayer Sigmoid 10 5 
      input <- normalArray 0 0.5 (Z:.10)
      let accVec = feedForward layer (param layer) input
      accVec `shouldSatisfy` isInRange 0 1
  describe "Network" $ do
    (test, train) <- runIO $ loadMnistArrays "/usr/local/share/mnist"
    let usedTest = Prelude.map (use *** constant) test
    let usedTrain = Prelude.map (use *** constant) train
    let flattenLayer = mkReshapeLayer (index2 28 28 :: Exp DIM2) (index1 (28*28) :: Exp DIM1) 
    let normLayer = mkNormalizeLayer 255 (index1 (28*28)) 
    fcl <- runIO $ mkFullyConnectedLayer Sigmoid (28*28) (10) 
    let net = Network (flattenLayer <!> normLayer <!> fcl) CrossEntropy  

    it "should run" $ do
      layer <- mkFullyConnectedLayer Sigmoid 10 5
      input <- mapM (const $ normalArray 0 0.5 (Z:.10)) [1..10]
      let net = Network layer CrossEntropy
      let accVec = netFeedForward net input
      accVec `shouldSatisfy` allInRange 0 1
    it "should run with mnist" $ do
      let inputs = Prelude.map Prelude.fst usedTest
      let accVec = netFeedForward net inputs
      accVec `shouldSatisfy` allInRange 0 1
    it "should run sgd as well" $ do
      (newNet, info) <- netSGD net usedTrain usedTest 2 100 0.5 0.5

      print (param (innerLayer newNet))
      print info
      False `shouldBe` True
      
    

