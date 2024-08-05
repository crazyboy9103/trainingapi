
https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.or5z4c2yi1qo 참고

To add a new Op:

1. Create a new directory
2. Implement new ops there
3. Delcare its Python interface in `vision.cpp`.
4. Compile the new ops by `python setup.py install`
5. Define its Python interface (+ torch.autograd.Function) in `layers/*.py`.
