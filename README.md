# llmamd - LLM assisted model development

This project was created to test the hypothesis that an LLM can be used to enhance the performance of a non-transformer-based text classifier.  This is done by using a Hugging Face model (HFM) to generate text that augments the original training data.

The HFM is used to generate the number of samples that are originally present in the training data effectively doubling the training samples.  The HFM is prompted to generate text samples in two specific ways referred to as *uniformed assistance* and *informed assistance* which are defined as follows:

+ Uniformed assistance - This type of assistance simply asks the HFM to create similar text to each sample in the training data.  The following prompt was used to generate these samples:  **PROMPT:**  *prompt tbd*
+ Informed assistance - This type of assistance asks the HFM to create similar text, but also to utilize one or more of the top x (tbd) words found in the target class samples (disaster-related) when creating the generated text.  The following prompt was used to generate these samples:  **PROMPT:**  *prompt tbd*


