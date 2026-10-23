# Food Recommendation System Using ChromaDB and RAG

## 1. Project overview

This project uses a food dataset to recommend dishes based on a user's preferences. It combines vector similarity search, structured metadata filters, and a language model that explains the recommendations.

These notes document the supplied lab overview and explain the concepts using illustrative examples. They are not a reproduction of the complete official lab or a report of completed implementation. The linked lab instructions could not be retrieved while preparing this guide.

## 2. What the project demonstrates

| Feature | Purpose | Example request |
| --- | --- | --- |
| Interactive CLI search | Accept requests in the terminal and display matching foods | “A sweet dessert with a buttery crust” |
| Filtered similarity search | Combine meaning-based search with exact conditions | “A sweet American dessert under 350 calories” |
| RAG chatbot | Retrieve relevant foods and use an LLM to generate an explanation | “Recommend a light dinner and explain why it fits” |

The CLI is the interface. Similarity search and filtering select records. RAG adds answer generation on top of retrieval.

## 3. Main components

| Component | Responsibility |
| --- | --- |
| Food dataset | Supplies descriptions, ingredients, nutrition, cooking methods, and taste profiles |
| Embedding model | Converts food descriptions and search questions into vectors |
| ChromaDB | Stores records and vectors, applies filters, and retrieves similar records |
| Retriever | Provides the retrieval step used by an answer-generation workflow |
| LLM | Generates a natural-language response using retrieved information |
| CLI | Reads user input and prints results or answers |

The embedding model and the LLM have different jobs. Embeddings support searching; the LLM produces the final explanation.

## 4. Dataset structure

The supplied dataset example is:

```json
{
    "food_id": 1,
    "food_name": "Apple Pie",
    "food_description": "A classic dessert made with a buttery, flaky crust filled with tender, spiced apples.",
    "food_calories_per_serving": 320,
    "food_nutritional_factors": {
        "carbohydrates": "42g",
        "protein": "2g",
        "fat": "16g"
    },
    "food_ingredients": ["Apples", "Flour", "Butter", "Sugar", "Cinnamon", "Nutmeg"],
    "food_health_benefits": "Rich in antioxidants and dietary fiber",
    "cooking_method": "Baking",
    "cuisine_type": "American",
    "food_features": {
        "taste": "sweet",
        "texture": "crisp and tender",
        "appearance": "golden brown",
        "preparation": "baked",
        "serving_type": "hot"
    }
}
```

Each food becomes one Chroma record. The dataset dictionary must be transformed into a string ID, document text, and flat metadata.

| Dataset field | Chroma representation |
| --- | --- |
| `food_id` | String record ID, such as `"1"` |
| Name and description | Searchable document text; name also stored in metadata |
| Ingredients | Joined into document text and a display string in metadata |
| Calories per serving | Numeric `calories` metadata |
| Nutritional factors | Numeric `carbohydrates_g`, `protein_g`, and `fat_g` fields |
| Cuisine and cooking method | Both text and metadata |
| Nested food features | Both text and separate metadata fields |
| Health benefits | Stored as a dataset-provided claim, not independently verified advice |

Metadata supports filtering and display. It is not automatically embedded with document text. An attribute can appear in both deliberately: taste influences semantic matching in the document and can also be filtered exactly through metadata.

**Collection metadata** describes the collection as a whole. **Record metadata** describes an individual food.

## 5. Preparing and indexing the data

The following uses **native Chroma**, consistent with `collection.query()` below. Assume `foods` is a list of dictionaries following the supplied schema and `collection` already has an embedding function configured.

```python
ids = []
documents = []
metadatas = []

for food in foods:
    features = food["food_features"]
    nutrition = food["food_nutritional_factors"]
    ingredients = ", ".join(food["food_ingredients"])

    ids.append(str(food["food_id"]))

    documents.append(
        f"Name: {food['food_name']}. "
        f"Description: {food['food_description']} "
        f"Ingredients: {ingredients}. "
        f"Cuisine: {food['cuisine_type']}. "
        f"Cooking method: {food['cooking_method']}. "
        f"Taste: {features['taste']}. "
        f"Texture: {features['texture']}. "
        f"Appearance: {features['appearance']}. "
        f"Preparation: {features['preparation']}. "
        f"Serving type: {features['serving_type']}. "
        f"Calories per serving: {food['food_calories_per_serving']}. "
        f"Nutrition: carbohydrates {nutrition['carbohydrates']}, "
        f"protein {nutrition['protein']}, fat {nutrition['fat']}. "
        f"Dataset-reported health benefits: {food['food_health_benefits']}."
    )

    metadatas.append({
        "name": food["food_name"],
        "calories": food["food_calories_per_serving"],
        "carbohydrates_g": float(nutrition["carbohydrates"].removesuffix("g")),
        "protein_g": float(nutrition["protein"].removesuffix("g")),
        "fat_g": float(nutrition["fat"].removesuffix("g")),
        "ingredients": ingredients,
        "cuisine": food["cuisine_type"],
        "cooking_method": food["cooking_method"],
        "taste": features["taste"],
        "texture": features["texture"],
        "appearance": features["appearance"],
        "preparation": features["preparation"],
        "serving_type": features["serving_type"],
        "health_benefits_reported": food["food_health_benefits"],
    })

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)
```

For one example, set `foods = [food]`, where `food` contains the Apple Pie dictionary. For the complete dataset, use its list of food dictionaries.

The nutrition conversion assumes strings consistently use grams, such as `"42g"`. Validate missing values and differing units before processing the full dataset. The example states calories per serving; confirm the basis for macronutrients in the dataset documentation.

For Apple Pie, the stored metadata includes `calories=320`, `carbohydrates_g=42.0`, `protein_g=2.0`, and `fat_g=16.0`.

`ids[0]`, `documents[0]`, and `metadatas[0]` describe the same food. The ID is its database identity; the list position only associates values within this call.

Chroma generates vectors from the supplied document text using the configured embedding function. Short records can remain one document per dish; they do not need a splitter merely because this is a RAG application.

Prepare and ingest the dataset once, then reuse the collection for questions. Use stable IDs and choose add, update, or upsert intentionally when reprocessing records.

## 6. Approach 1: similarity search

Example request:

> I want a sweet dessert with a buttery crust.

The system embeds the request and searches for nearby food vectors. It can find related descriptions even when they do not use exactly the same words.

```python
results = collection.query(
    query_texts=["A sweet dessert with a buttery crust"],
    n_results=3,
)
```

This returns relevant records, not an LLM-generated answer.

### Reading results

```python
for record_id, document, metadata, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(record_id)
    print(metadata["name"])
    print(document)
    print("Distance:", distance)
```

The first `[0]` selects the results for the first input query. Within that query's results, the lists align by position.

### Interpreting distance

For cosine distance, lower values mean closer vector directions:

```text
cosine distance = 1 - cosine similarity
```

A distance is not a probability or a guaranteed percentage match. Interpret scores according to the collection's configured metric. Returning the top three records also does not guarantee that all three are relevant.

## 7. Approach 2: metadata filtering

Similarity search captures meaning. Metadata filters enforce explicit conditions.

Example request:

> Find a sweet American dessert under 350 calories.

| Request part | Mechanism |
| --- | --- |
| Sweet dessert | Semantic similarity search |
| American cuisine | Exact condition on `cuisine` |
| Under 350 calories | Numeric condition on `calories` |

The filter narrows the eligible records, and similarity search ranks eligible matches. If no records satisfy the conditions, the system should say so rather than silently relax them.

Store numeric fields as numbers: `"calories": 350`, not `"calories": "350"`. Keep units and serving sizes consistent across the dataset.

Metadata filtering and document filtering are different:

- **Metadata filtering:** a structured field satisfies a condition, such as cuisine type.
- **Document filtering:** the stored text satisfies a content condition, such as containing a word.
- **Similarity search:** vectors are ranked by semantic closeness.

The supplied schema has no explicit vegetarian or allergen-safety field. Do not assume those flags already exist. Add validated fields before supporting strict dietary filters; ingredients alone may not establish preparation or cross-contact details.

## 8. Interactive CLI

A command-line interface runs in the terminal. It repeatedly asks for input and displays results.

```python
while True:
    question = input("Describe your meal, or type 'exit': ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    results = collection.query(
        query_texts=[question],
        n_results=3,
    )

    for text in results["documents"][0]:
        print(text)
```

Initialize models and the collection outside the loop. The loop should reuse them rather than rebuild the database for every question.

## 9. Approach 3: RAG chatbot

RAG means **Retrieval-Augmented Generation**. The application retrieves relevant information and supplies it to an LLM as context.

1. Receive the user's food request.
2. Apply any explicit dietary or nutritional conditions.
3. Retrieve matching foods from Chroma.
4. Format their descriptions and relevant metadata into context.
5. Send the context and question to the LLM.
6. Display an answer with the recommended food names and supporting details.

### Example prompt structure

```text
Recommend foods using only the supplied records.
Respect the explicit constraints in the request.
Do not invent ingredients or nutritional values.
If no suitable food is provided, explain that no match was found.

Food records:
{context}

User request:
{question}

Recommendation:
```

Include required nutrition and other metadata in the context explicitly. A framework that combines only document text will not automatically expose every metadata field to the LLM.

The LLM explains the retrieved choices. It does not replace the vector database or enforce the database filters on its own.

## 10. Connecting this to the earlier LangChain example

Your Acme policy example has the same structure:

| Earlier example | Food project |
| --- | --- |
| Policy documents | Food descriptions |
| Source metadata | Food names, nutrition, and dietary attributes |
| Question about remote work | Request for a meal |
| Retrieved policy context | Retrieved food context |
| LLM policy answer | LLM food recommendation |

To avoid mixing interfaces, identify which API an example uses:

| Interface | Typical object | Retrieval call |
| --- | --- | --- |
| Native Chroma | `collection` | `collection.query(...)` |
| LangChain with Chroma | `vectorstore` | `vectorstore.similarity_search(...)` |
| LangChain retriever | `retriever` | `retriever.invoke(...)` |

These are different interfaces to the retrieval task. Native Chroma returns a dictionary of result lists; LangChain typically returns `Document` objects.

Your existing `RetrievalQA` example is a legacy LangChain API. It remains useful for understanding the retrieve–format–generate sequence; package imports and supported APIs depend on the installed version.

## 11. Conversation history

A loop that accepts several questions is interactive, but it does not automatically remember previous requests.

Example:

```text
User: Recommend an American dessert.
User: Can you suggest one with more protein?
```

The second request depends on the first. A conversational implementation must retain relevant history or structured preferences, then use them when constructing the next retrieval request. Keep explicit constraints, such as cuisine type, active until the user changes them.

## 12. Implementation checklist

- [ ] Load and inspect the actual food dataset.
- [ ] Confirm field names, nutrition units, and serving sizes.
- [ ] Prepare stable IDs, descriptions, and metadata.
- [ ] Configure an embedding model and create the collection.
- [ ] Ingest the records once.
- [ ] Implement similarity search and display distances.
- [ ] Add structured filtering.
- [ ] Handle empty queries and no matching records.
- [ ] Build the terminal interaction loop.
- [ ] Connect retrieved context to an LLM.
- [ ] Display source food records with recommendations.
- [ ] Add conversation state if follow-up questions are supported.

## 13. Suggested checks

| Check | Expected behavior |
| --- | --- |
| Search for a known dish | Relevant records appear near the top |
| Use a paraphrase | Related foods can still be found |
| Request American cuisine | Every returned record has `cuisine` equal to `American` |
| Set a calorie limit | Every returned record satisfies the numeric condition |
| Use impossible conditions | Report that no match was found |
| Ask for missing nutrition information | Avoid inventing a value |
| Ask a follow-up question | Preserve earlier constraints if conversation state is implemented |

## 14. Sources and scope

- User-supplied IBM Skills Network / Coursera lab overview and Apple Pie dataset record.
- [Official lab instructions](https://cf-courses-data.static.labs.skills.network/SCOKX2aWllo4lzXVimQfqg/Food%20recommendation%20system%20using%20ChromaDB%20and%20RAG-v1.md.html) — link supplied with the overview; full contents were unavailable during preparation.
- [Chroma usage guide](https://github.com/chroma-core/docs/blob/main/docs/usage-guide.md) — record insertion, embeddings, metadata, and query concepts.
- [LangChain Chroma reference](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) — wrapper interface.

Code snippets are instructional fragments and assume their referenced objects already exist. This guide does not claim that the lab has been implemented or executed.
