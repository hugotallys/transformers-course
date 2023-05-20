import sys
import numpy as np

from transformers import pipeline

if __name__ == "__main__":

    try:
        taskname = sys.argv[1]
    except IndexError:
        print("Usage: transformers_overview.py <taskname>")
        sys.exit(1)

    if taskname == "sa":
        print("Sentiment Analysis\n")

        classifier = pipeline("sentiment-analysis")

        inputs = [
            "Fair price for the quality given. Very nice place!",
            "Nice ambience, but bad food and price doesn't match, too high. Lousy service.",
            "Extremely misleading, the name has rock and rock has nothing, absurd policy of not being able to share the glass and abusive prices, I was excited and lost the night.",
            "A pleasant surprise, the place is very top, from the service, the music and the food."
        ]

        outputs = classifier(inputs)

        for input, output in zip(inputs, outputs):
            print(f"\nInput:\t{input}")

            label = output["label"]

            if label == "POSITIVE":
                label += " 😃"
            else:
                label += " 😞"

            print("Output:\t %s with score %.2f\n" % (label, output["score"]))
    elif taskname == "zsc":
        print("Zero-Shot Classification\n")

        classifier = pipeline("zero-shot-classification")

        inputs = [
            "This is a course about the Transformers library",
            "Senators from the Democratic Party are pushing for extensive new federal voting rights legislation",
            "The company is looking foward to the next quarter results"
        ]
        candidate_labels = ["education 📖", "politics ⚖️", "business 📈"]

        outputs = classifier(inputs, candidate_labels)

        for input, output in zip(inputs, outputs):
            print(f"\nInput:\t{input}")

            labels = output["labels"]
            scores = output["scores"]

            max_score = np.argmax(scores)

            print(f"Output:\t The text is about {labels[max_score]}\n")
    elif taskname == "tg":
        print("Text Generation\n")

        generator = pipeline("text-generation")

        input_text = input("\n> ")

        output = generator(input_text, max_length=15, num_return_sequences=1)[0]

        print(f"\n ... {output['generated_text']}")
    else:
        print("Wrong taskname.")
        sys.exit(1)
