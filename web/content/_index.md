+++
title = "Home"
date = 2023-01-01T08:00:00-07:00
draft = false
[accordions]
[accordions.setup]
[[accordions.setup.items]]
title = "Structure Controls"
file = "setup/structure.html"
collapseId = "setupCollapseOne"
expanded = true
[[accordions.setup.items]]
title = "Advanced Controls"
file = "setup/advanced.html"
collapseId = "setupCollapseTwo"
expanded = false

[accordions.control]
[[accordions.control.items]]
title = "Load Your Data"
file = "control/data.html"
collapseId = "controlCollapseOne"
expanded = true
[[accordions.control.items]]
title = "Train It"
file = "control/train.html"
collapseId = "controlCollapseTwo"
expanded = false
[[accordions.control.items]]
title = "Test It"
file = "control/test.html"
collapseId = "controlCollapseThree"
expanded = false
+++

# Set up The Neural Network

{{< include-script "wasm/main.js" >}}
{{< accordion "setup" >}}

{{< include-script "js/visualizer.js" >}}
<div id="visualizer"></div>

---

# It's Super Fast, Give it a try!

{{< include-script "js/table.js" >}}
{{< include-script "js/io.js" >}}
{{< accordion "control" >}}

---