{-# LANGUAGE TypeOperators #-}
import Criterion.Main
import Control.Monad
import Control.Applicative
import Data.Array.Accelerate.NeuralNet
import Data.Array.Accelerate as A
import Data.Array.Accelerate.CUDA as I

benchNum f n = bench (show n) $ whnf f n
benchNumIO f n = bench (show n) $ whnfIO $ f n

benchNum2 :: (Show a, Show b) => (a -> b -> c) -> a -> b -> Benchmark
benchNum2 f x y = bench (show x Prelude.++ "x" Prelude.++ show y) $ whnf (f x) y

actApp a n = I.run1 (act_apply a) (A.fromList (Z:.n) [-20.0, -19.5..])
actDelt a n = I.run1 (act_delta a) (A.fromList (Z:.n) [-20.0, -19.5..])

f2n n = I.run1 (act_apply Sigmoid) (A.fromList (Z:.2 * n) [-20.0, -19.5 ..])

app f n = I.run1 f (A.fromList (Z:.n) [-1.0,-0.9..])

ffEnv = do
  l <- mkNormalMatMulLayer 0 1 10 15
  return (feedForward l (param l))


run2 :: (Arrays a, Arrays b, Arrays c) => (Acc a -> Acc b -> Acc c) -> a -> b -> c
run2 f x y = run (f (use x) (use y)) 

-- | This one keeps z on the GPU and runs `use` on all of the xs
foldRun :: (Arrays a, Arrays b) => (Acc a -> Acc b -> Acc a) -> a -> [b] -> a
foldRun f z xs = run (foldl f (use z) $ Prelude.map use xs)

-- | This one takes z from the GPU to CPU and back each loop
foldRun' :: (Arrays a, Arrays b) => (Acc a -> Acc b -> Acc a) -> a -> [b] -> a
foldRun' f z [] = z
foldRun' f z (x:xs) = foldRun f (run2 f z x) xs

z vecLen = A.fromList (Z:.(vecLen::Int)) $ Prelude.repeat (100::Float)
xs n vecLen = Prelude.replicate n (A.fromList (Z:.(vecLen::Int)) $ Prelude.repeat (1::Float)) 
foldTest f n v = f (A.zipWith (-)) (z v) (xs n v)

slicedFold :: (Elt e)
           => (a -> Acc (Array DIM1 e) -> a)
           -> a 
           -> Acc (Array (DIM1 :. Int) e) 
           -> a
slicedFold f z xs = foldl each z $ [0..len-1]
  where (_:.len) = unlift $ shape xs :: Exp DIM1 :. Exp Int
        each z n = f z $ slice xs $ lift (Z :. All :. n)

sliceTest f n v = run $ f (A.zipWith (-)) (use $ z v) (fill (lift (Z:.v:.n) :: Exp DIM2) 1)

main = defaultMain [
  -- So I used this test to figure out the overhead per `use` call and maybe the `run` function
  -- So I did the math and found that the overhead of the `run1` call is about 58 micro seconds.
  -- This doesnt seem like a lot but it can really pile up when `run1`ing 60,000 tests
  -- especialy if this is happening 10-100 times each this can take 100 minutes not taking into
  -- account the processing time here which should be the bulk of it.
  -- can I benchmark the difference between 1 run1 with Z:.100 vs 2 run1s with Z:.50
  -- So is f(2n) == 2f(n)?
    -- bgroup "Sigmoid calibration" $ Prelude.map ((benchNum sigmoidRun) . (*100)) [1 :: Int ..5]
    -- bgroup "f(2n)" $ Prelude.map ((benchNum f2n) . (*1000)) [1 :: Int .. 5]
  -- This proves 2f(n) /= f(2n)
  -- , bgroup "f(n)" $ Prelude.map ((benchNum sigmoidRun) . (*1000)) [1 :: Int .. 5]
    bgroup "activation" [
        bgroup "Sigmoid" [
            bgroup "act_apply" $ Prelude.map (benchNum $ actApp Sigmoid) [1000 :: Int]
          , bgroup "act_delta" $ Prelude.map (benchNum $ actDelt Sigmoid) [1000 :: Int]
        ]
      , bgroup "ReLU" [
            bgroup "act_apply" $ Prelude.map (benchNum $ actDelt ReLU) [1000 :: Int]
          , bgroup "act_delta" $ Prelude.map (benchNum $ actDelt ReLU) [1000 :: Int]
        ]
    ]
  , bgroup "loadMnist" [
        bench "one arr" $ nfIO (toList . Prelude.fst . head . Prelude.fst<$>loadMnistArrays "/usr/local/share/mnist")
      , bench "100 test cases" $ nfIO (Prelude.map (toList.Prelude.fst) . Prelude.take 100 . Prelude.fst<$>loadMnistArrays "/usr/local/share/mnist")
      , bench "1000 test cases" $ nfIO (Prelude.map (toList.Prelude.fst) . Prelude.take 1000 . Prelude.fst<$>loadMnistArrays "/usr/local/share/mnist")
      , bench "all test cases" $ nfIO (Prelude.map (toList.Prelude.fst) . Prelude.fst<$>loadMnistArrays "/usr/local/share/mnist")
    ]
  , bgroup "mkNormalMatMulLayer" $ Prelude.map (benchNumIO (join $ mkNormalMatMulLayer 0 1)) [13::Int]
  , bgroup "foldRun" $ 
      (benchNum2 $ foldTest foldRun) <$> [10,20] <*> [100]
  , bgroup "foldRun'" $ 
      (benchNum2 $ foldTest foldRun') <$> [10,20] <*> [100]
  , bgroup "slicedFold" $ 
      (benchNum2 $ sliceTest slicedFold) <$> [10 :: Int ,20] <*> [100 :: Int ,200]
  ]
