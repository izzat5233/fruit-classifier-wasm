+++
title = "Home"
date = 2023-01-01T08:00:00-07:00
draft = false
[accordions]
[accordions.setup]
[[accordions.setup.items]]
title = "Structure Setup"
file = "setup/structure.html"
collapseId = "setupCollapseOne"
expanded = true
[[accordions.setup.items]]
title = "Advanced Settings"
file = "setup/advanced.html"
collapseId = "setupCollapseTwo"
expanded = false

[accordions.control]
[[accordions.control.items]]
title = "Data Preparation"
file = "control/data.html"
collapseId = "controlCollapseOne"
expanded = true
[[accordions.control.items]]
title = "Training Your Module"
file = "control/train.html"
collapseId = "controlCollapseTwo"
expanded = false
[[accordions.control.items]]
title = "Testing & Evaluation"
file = "control/test.html"
collapseId = "controlCollapseThree"
expanded = false
+++

{{< include-script "wasm/main.js" >}}
{{< include-html "initialize.html" >}}

{{< include-script "js/data.js" >}}
{{< include-script "js/io.js" >}}

# Configure Your Neural Network

<br>

- Before diving into data classification, let's set up your neural network. Our setup section is designed to guide you
  through the process of configuring the basic structure of your network. Adjust the layers, nodes, and more.

- **Important:** changing any option in this section will reset the network weights.

{{< accordion "setup" >}}

{{< include-script "js/visualizer.js" >}}
<div id="visualizer"></div>

---

# It's Super Fast, Give it a Try!

<br>

- Now that your neural network is configured, let’s put it to the test. Upload your data, train the network, and then
  evaluate its performance.

{{< include-script "js/TableObject.js" >}}
{{< include-script "js/table.js" >}}
{{< accordion "control" >}}

---