## Goal: To identify the intentional organization structures of the enron social network and to be able to decipher the complex social hierarchy from network graph analysis.

On neighbourhood degree sequences of complex networks - Kevin M. Smith

# Neighborhood Degree Sequences - Give us an idea about the local structure of a node
We assume that number of connections is directly proportional to importance of a node in the network in the context of a organizational hierarchy. (This assumption is not true for all social networks). Hence, we can fairly assume that NDS is a fair representation of the local strucuture around a node.

# Params Used:
1. Node Heterogenity
Measure of variance/spread in the degrees of a node's neighborhood in the network on average.
If the spread is less, the neighboring nodes all have similar degrees (or) if the spread is high, the neighboring nodes have varying degrees irrespective of the degree of the node around which the spread is measured. 

2. Neighborhood Similarity (S)
Proportion of nodes in the network whose NDS matches that of another node in the graph.
S is always between 0 and 1. This metric gives us an idea on the proportion of nodes in the network with a not unique local structure around them i.e this metric can answer the question, are there any repetitions in the local structures around nodes.

3. Heirarchial Complexity
Measures the variation in neighbourhood structures among nodes of the same degree.

4. Network Organization
Measures how structured and organized the network is beyond randomness.

We compare the network graph with random graphs with same degree distribution because we wish to answer the question

* Are repeated neighborhood patterns intentional structure or just mathematical consequences of degrees?

There is a chance that there is no intention behind the existence of a unique looking structure owing to randomness. We wish to isolate and avoid drawing any conclusions that nudge us in the direction where we mistake randomness for organisation.

Observed Metric = degree-driven effects(captured by CM) + true organization(what we want)

# Configurational Models and Wilcoxon signed-rank test. 

With truncation of the feature vector or NDS to 20 degrees and 10 config models. 

--- REAL NETWORK ---
Saved → enron_features_full.csv
{'Vn': np.float64(24931.977670881934), 'S': np.float64(0.5603401286383953), 'Omega': np.float64(0.022289456814088893), 'R': np.float64(145204.38720775157), 'R_omega': np.float64(145201.1660971977)}

--- CONFIGURATION MODELS ---
Run 1/10
Run 2/10
Run 3/10
Run 4/10
Run 5/10
Run 6/10
Run 7/10
Run 8/10
Run 9/10
Run 10/10

--- COMPARISON ---
Vn:
Real: 24931.97767
CM mean: 15024.46580
p-value: 1.95312e-03

S:
Real: 0.56034
CM mean: 0.34372
p-value: 1.95312e-03

Omega:
Real: 0.02229
CM mean: 0.00671
p-value: 1.95312e-03

R:
Real: 145204.38721
CM mean: 31112.81115
p-value: 1.95312e-03

R_omega:
Real: 145201.16610
CM mean: 31112.75320
p-value: 1.95312e-03

At 95% confidence interval, all the metrics reject the null hypothesis that real = cm mean, hence it can be said with 95% confidence that none of the metric the CM mean. Hence there is more information in the graph that doesn't arise from randomness alone.

# Using Principal Component Analysis. 
The feature vector for a node is the 20 neighbor nodes with highest degrees. Using PCA on the data gives us 20 transformed dimensions with approximately 80% variance explained by the first component PC1 (pca_variance_explained.csv)

Influence Scoring for a node:
influence = alpha * degree + (1-alpha) * PCA1

degree - directly factors on the influence of a node

PCA1 captures the percentage of variation captured on the axis in the local structure (NDS) - it is a measure of how highly connected the neighboring nodes of a node is and hence, playing a role in the influence a node holds.

The top 10 nodes with most influence are:
Node,   PC1,                Degree,     influence_score
273,    15.630801900593204, 1367,       4.648625797026574
136,    16.343684362965224, 1026,       4.637095283835498
1028,   15.783899855229082, 1244,       4.632703819117593
370,    16.107140170925376, 1099,       4.630019537728157
140,    15.756635517885885, 1245,       4.628958702192976
458,    15.701687095995492, 1261,       4.626000142891262
195,    15.856335268161317, 1143,       4.608681748042489
1139,   15.62554621684311,  1068,       4.5462487738122235
566,    15.977418424706109, 924,        4.539252111152948

Influence score model used take into account influence due to direct connections and indirect connections due to connections of your neighbors. 
What this model fails to take into account when measuring influence is how well connected. 
