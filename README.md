
# **Author Verification Using Siamese Neural Networks**

This project employs a Siamese Neural Network (SNN) to verify authorship by analyzing stylistic features in text samples. It utilizes the **Gungor 2018 Victorian Author Attribution** dataset to train and evaluate the model.

## **Project Overview**

The goal of this project is to determine whether two text samples were written by the same author. Using deep learning techniques, the model captures stylistic differences and similarities to achieve high accuracy in authorship verification tasks.

### **Key Features**
- Uses character embeddings for tokenizing text.
- Employs convolutional layers to extract low-level features.
- Integrates an LSTM layer to capture long-term dependencies in writing style.
- Calculates similarity using Euclidean distance.

## **Dataset**
### **Source**
The dataset used for this project is **Gungor_2018_VictorianAuthorAttribution_data-train**, available on Kaggle:
- [Kaggle Dataset Link](https://www.kaggle.com/competitions/victorian-author-attribution/data)

### **Details**
- Text samples from 50 authors with standardized lengths for consistent analysis.
- For this project, we worked with **10 authors** (not the full dataset) to demonstrate the model’s capabilities effectively.


## **Installation**

### **Prerequisites**
- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Jupyter Notebook or Google Colab

### **Steps to Set Up**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/AuthorVerification.git
   cd AuthorVerification
   ```

2. **Install Dependencies**
   Install all required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**

### **Step 1: Dataset Setup**
1. Download the dataset from Kaggle and upload `Gungor_2018_VictorianAuthorAttribution_data-train.csv` to your Google Drive.
2. Mount your Google Drive in the Jupyter Notebook or Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Define the dataset path:
   ```python
   dataset_path = '/content/drive/MyDrive/Gungor_2018_VictorianAuthorAttribution_data-train.csv'
   ```

### **Step 2: Data Processing**
1. The dataset is preprocessed into paired sequences (`Xtrain`, `Xtest`, `Ytrain`, `Ytest`).
2. Run the preprocessing cells in `AuthorVerification.ipynb` to generate the `.npy` files:
   - `Xtrain.npy`, `Ytrain.npy` (Training data)
   - `Xtest.npy`, `Ytest.npy` (Testing data)

### **Step 3: Train the Model**
1. Run the training script in `AuthorVerification.ipynb`.
2. The model is saved as `lstm_model.h5` in the `models/` directory.

### **Step 4: Test the Model**
Use the provided function to evaluate similarity on custom text pairs:
```python
def predict_author_similarity(model, tokenizer, text1, text2, max_length=800):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Tokenize and pad the texts
    seq1 = tokenizer.texts_to_sequences([text1])[0]
    seq2 = tokenizer.texts_to_sequences([text2])[0]
    seq1 = pad_sequences([seq1], maxlen=max_length)
    seq2 = pad_sequences([seq2], maxlen=max_length)
    
    # Predict using the model
    prediction = model.predict([seq1, seq2], verbose=0)[0][0]  # Get the prediction (distance)
    
    if prediction > 0.5:
        return "Same author"
    else:
        return "Different authors"

# Example usage
result = predict_author_similarity(model, tokenizer, "Text1 content here", "Text2 content here")
print(result)

```

## **Files Structure**
- **`data/`**: Generated `.npy` files (`Xtrain.npy`, `Ytrain.npy`, `Xtest.npy`, `Ytest.npy`) after preprocessing the dataset. These are created after loading the dataset and running the preprocessing code.
- **`models/lstm_model.h5`**: The trained model file, generated after training the model on 10 authors from the dataset.
- **`notebooks/AuthorVerification.ipynb`**: Main notebook for data preprocessing, training, and testing.
- **`reports/`**: Documentation and presentations related to the project.

## **Results**
- The model was trained on **10 authors** from the dataset, with 20 text samples per author.
- Training accuracy exceeded **90%**, showcasing the model's effectiveness in distinguishing authors' writing styles.

## **References**
- **Dataset**: Gungor_2018_VictorianAuthorAttribution_data-train (Kaggle)
- **Papers**:
  - "Siamese Convolutional Neural Networks for Authorship Verification"
  - "A Robust Approach to Authorship Verification Using Siamese Deep Learning"

