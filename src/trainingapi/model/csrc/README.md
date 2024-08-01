

To add a new Op:

1. Create a new directory
2. Implement new ops there
3. Delcare its Python interface in `vision.cpp`.
4. Compile the new ops by `python setup.py install`
5. Define its Python interface (+ torch.autograd.Function) in `layers/*.py`.
