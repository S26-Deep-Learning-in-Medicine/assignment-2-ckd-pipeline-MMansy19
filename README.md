[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ENVI9wlT)
# Assignment: Chronic Kidney Disease Diagnostic Pipeline

* **Dataset**: Chronic Kidney Disease Dataset (Kaggle)
* **URL**: `https://www.kaggle.com/datasets/mansoordaku/ckdisease`
* **Clinical Context**: You are working with a highly constrained dataset ($n=400$). An over-parameterised model will memorise the training data rather than learning generalised markers for CKD.

## Tasks
1. **Data Pipeline**: Load the CSV. To simplify this assignment, drop all categorical (string) columns and keep only the numerical clinical features. Handle any missing values (`NaN`s), scale the features appropriately to prevent data leakage, and perform an 80/20 train/validation split.
2. **The Engineered Failure**: Build a baseline Dense network. Train it specifically to fail. Generate and plot the training and validation loss curves to visually prove the model has overfitted (look for the U-shape).
3. **The Clinical Solution**: Build a second, engineered model. Deploy the architectural constraints and regularisation techniques learned in the tutorial to prevent the network from memorising the small cohort.
4. **Evaluation**: Compare the performance of both models. Accuracy is not the priority; you must optimise for the `Recall` metric. Missing a CKD diagnosis is clinically unacceptable.

## Submission Requirements
Submit a GitHub repository link containing your `.ipynb` file. The notebook must execute top-to-bottom without errors and include the following:
* **Visual Proof**: A plot showing the training/validation curves of both models side-by-side, demonstrating the reduction in overfitting.
* **Confusion Matrices**: Rendered matrices for both models proving a reduction in False Negatives.
* **Clinical Benchmark**: The engineered model must achieve >80% Recall on the validation set.
* **Mathematical Justification**: In a Markdown cell at the end of the notebook, print your final parameter count. Write exactly one sentence explaining why this parameter capacity, combined with your chosen regularisation strategy, mathematically prevents memorisation on a 400-patient dataset.

**Deadline:** Monday, March 2nd.
