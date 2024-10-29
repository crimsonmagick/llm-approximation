from transformers import pipeline

generator = pipeline("text-generation", device="cuda", num_return_sequences=2, max_length=512)
print(generator("In this course, we will teach you how to"))
