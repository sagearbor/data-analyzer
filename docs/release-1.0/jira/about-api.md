The About page should let a developer call the tool from code with no effort: blocks they can copy, paste and run.

# Acceptance criteria
1. The About page has copy-and-paste blocks for the common calls: check the service is up, analyze a file, analyze data sent as JSON, in both curl and Python.
2. Each block runs as written once the developer has set their key. A test runs the blocks exactly as shown on the page and checks the answer.
3. No real key ever appears on the page; the blocks read the key from the developer's own environment.
4. The page shows the correct address of the service for the environment it is running in.
5. The page says how to get a key and where the full API reference is.
