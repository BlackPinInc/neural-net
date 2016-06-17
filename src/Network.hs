module Network where

import Data.Matrix
import Data.Maybe (isJust, fromJust)

data Network = Network { numLayers :: Int
                       , sizes :: [Int]
                       , biases :: Matrix Double
                       , weights :: Matrix Double
                       } deriving (Show)
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

sigmoid :: Floating a => a -> a
sigmoid z = 1/(1 + exp(-1*z))

sigmoid' :: Floating a => a -> a
sigmoid' z = sigmoid(z)*(1-sigmoid(z))

-- feedForward :: Network -> Matrix Double -> Matrix Double
-- feedForward net a = reshape (last $ sizes net) vectorOut
  -- where vectorOut = foldl (\a (b,w) -> sigmoid $ w * a + b) (flatten a) $ zip (toColumns $ biases net) (toColumns $ weights net)

-- sgd :: Network -> [TrainingSet] -> Int -> Int -> Double -> (Maybe [TrainingSet]) -> IO Network
-- sgd net trainingData epochs miniBatchSize eta testData = 1
  -- where ntest = if isJust testData then fromJust $ length <$> testData else 0

-- def SGD(self, training_data, epochs, mini_batch_size, eta,
--             test_data=None):
--         if test_data: n_test = len(test_data)
--         n = len(training_data)
--         for j in xrange(epochs):
--             random.shuffle(training_data)
--             mini_batches = [
--                 training_data[k:k+mini_batch_size]
--                 for k in xrange(0, n, mini_batch_size)]
--             for mini_batch in mini_batches:
--                 self.update_mini_batch(mini_batch, eta)
--             if test_data:
--                 print "Epoch {0}: {1} / {2}".format(
--                     j, self.evaluate(test_data), n_test)
--             else:
--                 print "Epoch {0} complete".format(j)
