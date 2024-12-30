# Text Embedding

## Setting up environment
Start off by creating a virtual environment for your Python packages by running the followinig command:

```bash
python -m venv .venv
```
Activate your environment:

**Mac/Linux**:
```bash
source .venv/bin/activate
```

**Windows:**:
```bash
.venv\Scripts\activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running code

Once the virtual environment has been set up and activated, you can run the script:
```bash
python text-embedder.py
```

## Future improvements

For larger data sets, improvements can be made to scale the solution:

1. **Precompute and Store Embeddings**:
   - Generate embeddings for the corpus once and save them to disk using a format like `.npy` or `.hdf5`. This avoids recomputing them on every run.
   - Example:
     ```python
     import numpy as np
     np.save("corpus_embeddings.npy", corpus_embeddings)
     ```

2. **Use Approximate Nearest Neighbor (ANN) Search**:
   - For very large datasets, calculating cosine similarity for all documents becomes inefficient.
   - Use libraries like FAISS or Annoy for fast similarity searches.
   - Example with FAISS:
     ```bash
     pip install faiss-cpu
     ```
     ```python
     import faiss
     index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
     index.add(corpus_embeddings)
     _, indices = index.search(query_embedding, k=1)
     ```

3. **Batch Processing**:
   - If the corpus is too large to fit in memory, process it in smaller batches to avoid memory issues.

4. **Hardware Acceleration**:
   - Use GPUs to speed up the embedding generation process. Move the model and tensors to the GPU:
     ```python
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     model.to(device)
     ```