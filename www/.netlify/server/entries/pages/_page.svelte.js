import { b as get_store_value, c as create_ssr_component, a as subscribe, o as onDestroy, d as add_attribute, e as each, v as validate_component } from "../../chunks/ssr.js";
import { w as writable } from "../../chunks/index.js";
import * as d3 from "d3";
import { e as escape } from "../../chunks/escape.js";
const gameId = writable(null);
const viewportHeight = writable(0);
const states = writable(null);
const gameInfo = writable(null);
const currentStep = writable(0);
const intervalId = writable(null);
const scale = writable(null);
function createBackgroundGrid(svg, gridData, scale2) {
  if (!gridData || !scale2) return;
  const { solid, water, trees } = gridData;
  const cellSize = scale2(1) - scale2(0);
  const solidTileSize = cellSize * 0.5;
  const waterCircleRadius = cellSize * 0.05;
  const treeSize = cellSize * 0.1;
  const offset = (cellSize - solidTileSize) / 2;
  svg.selectAll(".background-cell").data(
    solid.flat().map((isSolid, index) => ({
      isSolid,
      x: index % solid[0].length,
      y: Math.floor(index / solid[0].length)
    }))
  ).join("rect").attr("class", "background-cell").attr("x", (d) => scale2(d.x)).attr("y", (d) => scale2(d.y)).attr("width", cellSize).attr("height", cellSize).attr("fill", "transparent");
  svg.selectAll(".solid-tile").data(
    solid.flat().map((isSolid, index) => ({
      isSolid,
      x: index % solid[0].length,
      y: Math.floor(index / solid[0].length)
    })).filter((d) => d.isSolid)
  ).join("rect").attr("class", "solid-tile ink").attr("x", (d) => scale2(d.x) + offset).attr("y", (d) => scale2(d.y) + offset).attr("width", solidTileSize).attr("height", solidTileSize).attr("fill", "#fff").attr("stroke", "none");
  svg.selectAll(".water-cell").data(
    water.flat().map((isWater, index) => ({
      isWater,
      x: index % water[0].length,
      y: Math.floor(index / water[0].length)
    })).filter((d) => d.isWater)
  ).join("circle").attr("class", "water-cell ink").attr("cx", (d) => scale2(d.x) + cellSize / 2).attr("cy", (d) => scale2(d.y) + cellSize / 2).attr("r", waterCircleRadius).attr("fill", "#fff").attr("stroke", "none");
  svg.selectAll(".tree-cell").data(
    trees.flat().map((isTree, index) => ({
      isTree,
      x: index % trees[0].length,
      y: Math.floor(index / trees[0].length)
    })).filter((d) => d.isTree)
  ).join("circle").attr("class", "tree-cell ink").attr("cx", (d) => scale2(d.x) + cellSize / 2).attr("cy", (d) => scale2(d.y) + cellSize / 2).attr("r", treeSize).attr("stroke", "none");
}
function getScaledPosition(d, currentScale) {
  return {
    x: currentScale ? currentScale(d.position[0]) : 0,
    y: currentScale ? currentScale(d.position[1]) : 0
  };
}
function createShape(d, x, y, radius) {
  switch (d.type) {
    case 0:
      return `M ${x},${y} m -${radius},0 a ${radius},${radius} 0 1,0 ${radius * 2},0 a ${radius},${radius} 0 1,0 -${radius * 2},0`;
    case 1:
      return `M ${x - radius},${y - radius} h ${radius * 2} v ${radius * 2} h -${radius * 2} z`;
    case 2:
      return `M ${x},${y - radius} L ${x - radius},${y + radius} L ${x + radius},${y + radius} Z`;
    default:
      return "";
  }
}
function positionHealthBar(d, currentScale) {
  const { x, y } = getScaledPosition(d, currentScale);
  const normalizedHealth = d.health / d.maxHealth;
  return {
    x: x - 5,
    y: y - 15,
    width: normalizedHealth * 10
    // Multiply by 10 to keep the same visual scale
  };
}
function calculateStreakPositions(agent, target, currentScale) {
  const start = getScaledPosition(agent, currentScale);
  const end = getScaledPosition(target, currentScale);
  const offsetRatio = 0.05;
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  const offsetX = dx / length * offsetRatio * length;
  const offsetY = dy / length * offsetRatio * length;
  return {
    x1: start.x + offsetX,
    y1: start.y + offsetY,
    x2: end.x - offsetX,
    y2: end.y - offsetY
  };
}
function updateVisualization(duration) {
  const currentStates = get_store_value(states);
  const currentScale = get_store_value(scale);
  const currentGameInfo = get_store_value(gameInfo);
  if (!currentStates || !currentScale || !currentGameInfo) return;
  console.log("Current GameInfo in updateVisualization:", currentGameInfo);
  const svg = d3.select("svg");
  createBackgroundGrid(svg, currentGameInfo.terrain, currentScale);
  const step = get_store_value(currentStep);
  const unitData = currentStates.unit_positions[step].map((position, i) => ({
    position,
    team: currentStates.unit_teams[step][i],
    type: currentStates.unit_types[step][i],
    health: currentStates.unit_health[step][i],
    maxHealth: currentGameInfo.unit_type_health[currentStates.unit_types[step][i]],
    attack: currentStates.prev_attack_actions[step][i]
  }));
  updateShapes(svg, unitData, currentScale, currentGameInfo, duration);
  updateHealthBars(svg, unitData, currentScale, duration);
  updateAttackStreaks(svg, unitData, currentScale, duration);
}
function updateShapes(svg, unitData, currentScale, gameInfo2, duration) {
  const shapes = svg.selectAll(".shape").data(unitData, (d, i) => i.toString());
  shapes.enter().append("path").attr("class", (d) => `shape ink type-${d.type} ${d.team === 0 ? "ally" : "enemy"}`).merge(shapes).transition().duration(duration).ease(d3.easeLinear).attr("d", (d) => {
    const { x, y } = getScaledPosition(d, currentScale);
    const radius = currentScale(gameInfo2.unit_type_radiuses[d.type]);
    return createShape(d, x, y, radius);
  });
  shapes.exit().remove();
}
function updateHealthBars(svg, unitData, currentScale, duration) {
  const healthBars = svg.selectAll(".health-bar").data(unitData, (d, i) => i.toString());
  healthBars.enter().append("rect").attr("class", "health-bar ink").merge(healthBars).transition().duration(duration).ease(d3.easeLinear).attr("x", (d) => positionHealthBar(d, currentScale).x).attr("y", (d) => positionHealthBar(d, currentScale).y).attr("width", (d) => positionHealthBar(d, currentScale).width).attr("height", 2);
  healthBars.exit().remove();
}
function updateAttackStreaks(svg, unitData, currentScale, duration) {
  svg.selectAll(".streak").remove();
  const team0Units = unitData.filter((u) => u.team === 0);
  const team1Units = unitData.filter((u) => u.team === 1);
  for (let i = 0; i < unitData.length; i++) {
    const agent = unitData[i];
    if (agent.attack >= 5) {
      let target;
      if (agent.team === 0) {
        const targetIndex = agent.attack - 5;
        target = team1Units[targetIndex];
      } else {
        const targetIndex = team0Units.length - 1 - (agent.attack - 5);
        target = team0Units[targetIndex];
      }
      if (target) {
        const { x1, y1, x2, y2 } = calculateStreakPositions(agent, target, currentScale);
        svg.append("line").attr("class", "streak ink").attr("x1", x1).attr("y1", y1).attr("x2", x1).attr("y2", y1).attr("stroke-width", 3).attr("stroke-opacity", 0.6).transition().duration(duration).ease(d3.easeLinear).attr("x2", x2).attr("y2", y2).attr("stroke-opacity", 0).remove();
      }
    }
  }
}
function handleResize() {
  if (typeof window !== "undefined") {
    const newVh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * 0.96;
    viewportHeight.set(newVh);
    const currentGameInfo = get_store_value(gameInfo);
    console.log("Current GameInfo:", currentGameInfo);
    if (!currentGameInfo || !currentGameInfo.terrain || !currentGameInfo.terrain.solid) {
      console.warn("GameInfo or terrain data is missing");
      return;
    }
    const gridSize = currentGameInfo.terrain.solid.length > 0 ? currentGameInfo.terrain.solid[0].length : 100;
    scale.set(d3.scaleLinear().domain([0, gridSize]).range([0, newVh]));
    updateVisualization(0);
  }
}
const css$2 = {
  code: "#simulation.svelte-1x8c203{padding:2vh;height:96vh;width:96vh}button.svelte-1x8c203{padding:10px 20px;font-size:16px}svg.svelte-1x8c203{height:100%;width:100%}",
  map: '{"version":3,"file":"Simulation.svelte","sources":["Simulation.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { onMount, onDestroy } from \\"svelte\\";\\nimport { gameId, states, gameInfo, intervalId } from \\"$lib/store\\";\\nimport { get } from \\"svelte/store\\";\\nimport { startAnimation, updateVisualization } from \\"$lib/plots\\";\\nimport { handleResize } from \\"$lib/utils\\";\\nasync function startGame() {\\n  const response = await fetch(\\"http://localhost:8000/run\\", {\\n    method: \\"POST\\"\\n  });\\n  const data = await response.json();\\n  gameId.set(data.game_id);\\n  states.set(data.states);\\n  gameInfo.set(data.env_info);\\n  console.log(data.env_info);\\n  startAnimation();\\n  updateVisualization(0);\\n}\\nonDestroy(() => {\\n  const currentIntervalId = get(intervalId);\\n  if (currentIntervalId !== null) {\\n    clearInterval(currentIntervalId);\\n  }\\n  if (typeof window !== \\"undefined\\") {\\n    window.removeEventListener(\\"resize\\", handleResize);\\n  }\\n});\\nonMount(() => {\\n  if (typeof window !== \\"undefined\\") {\\n    startGame().then(() => {\\n      handleResize();\\n      window.addEventListener(\\"resize\\", handleResize);\\n    });\\n  }\\n});\\n$: if (get(gameInfo)) {\\n  handleResize();\\n}\\n<\/script>\\n\\n<div id=\\"simulation\\">\\n    {#if !$gameId}\\n        <button on:click={startGame}>run</button>\\n    {:else}\\n        <svg />\\n    {/if}\\n</div>\\n\\n<style>\\n    #simulation {\\n        padding: 2vh;\\n        height: 96vh;\\n        width: 96vh;\\n    }\\n    button {\\n        padding: 10px 20px;\\n        font-size: 16px;\\n    }\\n    svg {\\n        /* border: 2px solid; */\\n        height: 100%;\\n        width: 100%;\\n    }\\n</style>\\n"],"names":[],"mappings":"AAgDI,0BAAY,CACR,OAAO,CAAE,GAAG,CACZ,MAAM,CAAE,IAAI,CACZ,KAAK,CAAE,IACX,CACA,qBAAO,CACH,OAAO,CAAE,IAAI,CAAC,IAAI,CAClB,SAAS,CAAE,IACf,CACA,kBAAI,CAEA,MAAM,CAAE,IAAI,CACZ,KAAK,CAAE,IACX"}'
};
const Simulation = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let $gameId, $$unsubscribe_gameId;
  $$unsubscribe_gameId = subscribe(gameId, (value) => $gameId = value);
  onDestroy(() => {
    const currentIntervalId = get_store_value(intervalId);
    if (currentIntervalId !== null) {
      clearInterval(currentIntervalId);
    }
    if (typeof window !== "undefined") {
      window.removeEventListener("resize", handleResize);
    }
  });
  $$result.css.add(css$2);
  {
    if (get_store_value(gameInfo)) {
      handleResize();
    }
  }
  $$unsubscribe_gameId();
  return `<div id="simulation" class="svelte-1x8c203">${!$gameId ? `<button class="svelte-1x8c203" data-svelte-h="svelte-givzr5">run</button>` : `<svg class="svelte-1x8c203"></svg>`} </div>`;
});
const css$1 = {
  code: '#controller.svelte-11c3l3i{height:96vh;margin:2vh 2vh 2vh 0}.bot.svelte-11c3l3i{padding:0.5rem;text-align:left;letter-spacing:0.08rem;line-height:3.5rem;text-align:justify;font-size:2rem;font-family:"IBM Plex Sans", sans-serif}.user.svelte-11c3l3i{padding:0.5rem;text-align:right}.history.svelte-11c3l3i{overflow-y:auto;height:calc(50% - 3rem)}',
  map: '{"version":3,"file":"Controller.svelte","sources":["Controller.svelte"],"sourcesContent":["<script lang=\\"ts\\">import Visualizer from \\"$lib/comps/Visualizer.svelte\\";\\nimport { afterUpdate } from \\"svelte\\";\\nlet history = [\\n  {\\n    content: `Welcome to the life like command and control (C2) simulator. You are in a C2 center, far removed from the war.`,\\n    author: \\"bot\\"\\n  },\\n  {\\n    content: `In this scenario, the war is global, and yet fought with guns. To do your part, you must act as a C2 commander of the allies.`,\\n    author: \\"bot\\"\\n  },\\n  {\\n    content: `You are fighting the enemies. Your troops can hide and move through trees, shoot but not cross water, while buildings are impenetrable.`,\\n    author: \\"bot\\"\\n  },\\n  {\\n    content: `Tell me what place on earth you would like to command. Kongens Have, Copenhagen, Denmark, could use your help.`,\\n    author: \\"bot\\"\\n  }\\n];\\nlet input;\\nfunction send() {\\n  history = [...history, { content: input.value, author: \\"user\\" }];\\n  input.value = \\"\\";\\n}\\nfunction handleKeydown(event) {\\n  if (event.key === \\"Enter\\") {\\n    send();\\n  }\\n}\\nlet historyContainer;\\nafterUpdate(() => {\\n  if (historyContainer) {\\n    const lastMessage = historyContainer.lastElementChild;\\n    if (lastMessage) {\\n      lastMessage.scrollIntoView({ behavior: \\"smooth\\" });\\n    }\\n  }\\n});\\n<\/script>\\n\\n<div id=\\"controller\\">\\n    <div class=\\"history\\" bind:this={historyContainer}>\\n        {#each history as message, i (message)}\\n            {#if message.author === \\"bot\\"}\\n                <div class=\\"bot\\">{message.content}</div>\\n            {:else}\\n                <div class=\\"user\\">{message.content}</div>\\n            {/if}\\n        {/each}\\n    </div>\\n\\n    <!-- <div class=\\"input\\">\\n        <input bind:this={input} type=\\"text\\" placeholder=\\"Type a message...\\" on:keydown={handleKeydown} />\\n    </div> -->\\n</div>\\n\\n<style>\\n    #controller {\\n        /*controler is 96vh high with 2 as padding above and bellow. Border is 2px solid*/\\n        height: 96vh;\\n        /* border: 2px solid; */\\n        margin: 2vh 2vh 2vh 0;\\n    }\\n    .bot {\\n        padding: 0.5rem;\\n        text-align: left;\\n        letter-spacing: 0.08rem;\\n        line-height: 3.5rem;\\n        text-align: justify;\\n        font-size: 2rem;\\n        font-family: \\"IBM Plex Sans\\", sans-serif;\\n    }\\n    .user {\\n        padding: 0.5rem;\\n        text-align: right;\\n    }\\n\\n    /* enable scroll through history */\\n    .history {\\n        overflow-y: auto;\\n        height: calc(50% - 3rem);\\n    }\\n\\n    /* visualizer is the top  half of the controler / screen  (50% height) */\\n    .input {\\n        /* put input at the bottom of the controler / screen */\\n        position: absolute;\\n        bottom: 0;\\n        display: flex;\\n        width: calc(100vw - 100vh);\\n    }\\n    input {\\n        width: 100%;\\n        height: 2rem;\\n        font-size: 1rem;\\n        padding: 0.5rem;\\n        background-color: black;\\n        color: white;\\n        border-radius: 0.5rem;\\n        border: solid 4px rgba(0, 0, 0, 0.1);\\n    }\\n</style>\\n"],"names":[],"mappings":"AA0DI,0BAAY,CAER,MAAM,CAAE,IAAI,CAEZ,MAAM,CAAE,GAAG,CAAC,GAAG,CAAC,GAAG,CAAC,CACxB,CACA,mBAAK,CACD,OAAO,CAAE,MAAM,CACf,UAAU,CAAE,IAAI,CAChB,cAAc,CAAE,OAAO,CACvB,WAAW,CAAE,MAAM,CACnB,UAAU,CAAE,OAAO,CACnB,SAAS,CAAE,IAAI,CACf,WAAW,CAAE,eAAe,CAAC,CAAC,UAClC,CACA,oBAAM,CACF,OAAO,CAAE,MAAM,CACf,UAAU,CAAE,KAChB,CAGA,uBAAS,CACL,UAAU,CAAE,IAAI,CAChB,MAAM,CAAE,KAAK,GAAG,CAAC,CAAC,CAAC,IAAI,CAC3B"}'
};
const Controller = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let history = [
    {
      content: `Welcome to the life like command and control (C2) simulator. You are in a C2 center, far removed from the war.`,
      author: "bot"
    },
    {
      content: `In this scenario, the war is global, and yet fought with guns. To do your part, you must act as a C2 commander of the allies.`,
      author: "bot"
    },
    {
      content: `You are fighting the enemies. Your troops can hide and move through trees, shoot but not cross water, while buildings are impenetrable.`,
      author: "bot"
    },
    {
      content: `Tell me what place on earth you would like to command. Kongens Have, Copenhagen, Denmark, could use your help.`,
      author: "bot"
    }
  ];
  $$result.css.add(css$1);
  return `<div id="controller" class="svelte-11c3l3i"><div class="history svelte-11c3l3i"${add_attribute()}>${each(history, (message, i) => {
    return `${message.author === "bot" ? `<div class="bot svelte-11c3l3i">${escape(message.content)}</div>` : `<div class="user svelte-11c3l3i">${escape(message.content)}</div>`}`;
  })}</div>  </div>`;
});
const css = {
  code: ".container.svelte-kymdx8{display:grid;grid-template-columns:100vh 1fr;grid-template-rows:1fr;width:100vw}#sim.svelte-kymdx8{grid-column:1;height:100%;width:100%}#con.svelte-kymdx8{grid-column:2}",
  map: '{"version":3,"file":"+page.svelte","sources":["+page.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { onMount } from \\"svelte\\";\\nimport Simulation from \\"$lib/comps/Simulation.svelte\\";\\nimport Controller from \\"$lib/comps/Controller.svelte\\";\\nimport Map from \\"$lib/comps/Map.svelte\\";\\nlet message = \\"Loading...\\";\\nlet error = null;\\nasync function fetchData() {\\n  try {\\n    const response = await fetch(\\"http://localhost:8000/\\");\\n    if (!response.ok) {\\n      throw new Error(`HTTP error! status: ${response.status}`);\\n    }\\n    const data = await response.json();\\n    message = data.Hello;\\n  } catch (e) {\\n    console.error(\\"There was a problem with the fetch operation: \\" + e.message);\\n    error = e.message;\\n  }\\n}\\nonMount(fetchData);\\n<\/script>\\n\\n<div class=\\"container\\">\\n    <!-- <div id=\\"map\\"><Map /></div> -->\\n    <div id=\\"sim\\"><Simulation /></div>\\n    <!-- <div id=\\"vis\\" class=\\"section\\"><Visualizer /></div> -->\\n    <div id=\\"con\\"><Controller /></div>\\n</div>\\n\\n<style>\\n    .container {\\n        display: grid;\\n        grid-template-columns: 100vh 1fr;\\n        grid-template-rows: 1fr;\\n        /* add spacing between the cells*/\\n        /* gap: 1rem; */\\n        /* height: calc(100vh - 4px); */\\n        width: 100vw;\\n    }\\n\\n    #sim {\\n        grid-column: 1;\\n        /* grid-row: 1 / span 2; */\\n        height: 100%;\\n        width: 100%;\\n    }\\n\\n    /* #vis {\\n        grid-column: 2;\\n        grid-row: 1;\\n    } */\\n\\n    #con {\\n        grid-column: 2;\\n    }\\n</style>\\n"],"names":[],"mappings":"AA8BI,wBAAW,CACP,OAAO,CAAE,IAAI,CACb,qBAAqB,CAAE,KAAK,CAAC,GAAG,CAChC,kBAAkB,CAAE,GAAG,CAIvB,KAAK,CAAE,KACX,CAEA,kBAAK,CACD,WAAW,CAAE,CAAC,CAEd,MAAM,CAAE,IAAI,CACZ,KAAK,CAAE,IACX,CAOA,kBAAK,CACD,WAAW,CAAE,CACjB"}'
};
const Page = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  $$result.css.add(css);
  return `<div class="container svelte-kymdx8"> <div id="sim" class="svelte-kymdx8">${validate_component(Simulation, "Simulation").$$render($$result, {}, {}, {})}</div>  <div id="con" class="svelte-kymdx8">${validate_component(Controller, "Controller").$$render($$result, {}, {}, {})}</div> </div>`;
});
export {
  Page as default
};
