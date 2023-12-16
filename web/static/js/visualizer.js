function drawNetwork(dimensions) {
    const svgWidth = d3.select("#visualizer").node().getBoundingClientRect().width;
    const radius = (window.innerWidth / 100) + (5000 / window.innerWidth);
    const svgHeight = (2 * radius) * (2 * Math.max(...dimensions));

    function layerYShift(nodesInLayer) {
        return (svgHeight - (nodesInLayer * 4 * radius)) / 2;
    }

    function layerXShift(layerIndex) {
        return (svgWidth / dimensions.length) * (layerIndex + 0.5) - 2 * radius;
    }

    function circleYShift(nodeIndex) {
        return radius * 2 * (2 * nodeIndex + 1);
    }

    // Create the SVG
    let svg = d3.select("#visualizer").select("svg");
    if (svg.empty()) svg = d3.select("#visualizer").append("svg");
    svg.attr("width", svgWidth).attr("height", svgHeight);

    const layerData = dimensions.map((n, i) => ({
        index: i,
        nodes: d3.range(n).map(j => ({layerIndex: i, nodeIndex: j}))
    }));

    // Create the layers
    const layers = svg.selectAll(".layer")
        .data(layerData)
        .join("g")
        .attr("class", "layer")
        .attr("transform", (d, i) =>
            `translate(${layerXShift(i)}, ${layerYShift(d.nodes.length)})`);

    // Draw the nodes
    layers.selectAll(".node")
        .data(d => d.nodes)
        .join("circle")
        .attr("class", "node")
        .attr("cx", radius * 2)
        .attr("cy", d => circleYShift(d.nodeIndex))
        .transition()
        .attr("r", radius);

    // Preparing data for lines
    const lineData = [];
    dimensions.forEach((numNodes, layerIndex) => {
        if (layerIndex === 0) return;

        const prevLayerNodes = layerData[layerIndex - 1].nodes;
        layerData[layerIndex].nodes.forEach((node, nodeIndex) => {
            prevLayerNodes.forEach((prevNode, prevNodeIndex) => {
                lineData.push({
                    x1: layerXShift(layerIndex - 1) + radius * 2,
                    y1: layerYShift(prevLayerNodes.length) + circleYShift(prevNodeIndex),
                    x2: layerXShift(layerIndex) + radius * 2,
                    y2: layerYShift(numNodes) + circleYShift(nodeIndex)
                });
            });
        });
    });

    // Drawing lines
    svg.selectAll(".line")
        .data(lineData)
        .join("line")
        .attr("class", "line")
        .attr("x1", d => d.x1)
        .attr("y1", d => d.y1)
        .attr("x2", d => d.x2)
        .attr("y2", d => d.y2)
        .attr("stroke", "black")
        .transition()
        .attr("stroke-width", 2)
}