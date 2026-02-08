**Entropy** - Is a measure of impurity of a data
**Information Gain** - Determines which feature to split on at each node

**Decision Tree Building Process:**
- Calculate information gain for all features at current node
- Select feature with highest information gain
- Split dataset into subsets based on selected feature
- Create left and right branches
- Route training examples to appropriate branches

**Recursive Splitting:**
- Repeat splitting process on left and right branches
- Continue until stopping criteria is met

**Stopping Criteria:**
- Node reaches 100% single class (entropy = 0)
- Tree exceeds maximum depth limit
- Information gain below threshold
- Number of examples in node below threshold

**One-Hot Encoding** 
- A technique to convert categorical features into binary vectors where each category is represented by a unique binary column with a value of 1 for the category present and 0 for all others