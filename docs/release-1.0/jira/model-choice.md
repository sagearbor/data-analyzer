Let the person choose a stronger AI model for reading data dictionaries, from a list the server allows.

# Acceptance criteria
1. The page shows a model list with more than one choice in the deployed environment. Today the deployed list has only one entry (gpt-5-nano).
2. The list is set on the server from the models actually available in Azure AI Foundry; a person cannot ask for a model outside it, on the page or through the API.
3. Each model on the list has been called for real and returned a usable answer. A model that fails is left off the list.
4. The output allowance for every model is at least 40,000 tokens, so models that think before answering do not return empty answers. Empty and cut-off answers are counted and reported.
5. The default stays the low-cost model; the page says which model is in use.
