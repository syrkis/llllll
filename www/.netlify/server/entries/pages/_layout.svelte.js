import { c as create_ssr_component } from "../../chunks/ssr.js";
const css = {
  code: ".ink{fill:none;stroke:black}.ink.enemy{fill:black;stroke:black}.ink.ally{fill:white;stroke:black}body{margin:0;overflow:hidden}@media(prefers-color-scheme: dark){.ink{fill:white;stroke:white}.ink.enemy{fill:white;stroke:white}.ink.ally{fill:black;stroke:white}body{background-color:black;color:white}}h1, h2{margin:0}",
  map: '{"version":3,"file":"+layout.svelte","sources":["+layout.svelte"],"sourcesContent":["<slot />\\n\\n<style>\\n    :global(.ink) {\\n        fill: none;\\n        stroke: black;\\n    }\\n    :global(.ink.enemy) {\\n        fill: black;\\n        stroke: black;\\n    }\\n    :global(.ink.ally) {\\n        fill: white;\\n        stroke: black;\\n    }\\n    :global(body) {\\n        margin: 0;\\n        overflow: hidden;\\n    }\\n\\n    @media (prefers-color-scheme: dark) {\\n        :global(.ink) {\\n            fill: white;\\n            stroke: white;\\n        }\\n        :global(.ink.enemy) {\\n            fill: white;\\n            stroke: white;\\n        }\\n        :global(.ink.ally) {\\n            fill: black;\\n            stroke: white;\\n        }\\n        :global(body) {\\n            background-color: black;\\n            color: white;\\n        }\\n    }\\n    :global(h1, h2) {\\n        margin: 0;\\n    }\\n</style>\\n"],"names":[],"mappings":"AAGY,IAAM,CACV,IAAI,CAAE,IAAI,CACV,MAAM,CAAE,KACZ,CACQ,UAAY,CAChB,IAAI,CAAE,KAAK,CACX,MAAM,CAAE,KACZ,CACQ,SAAW,CACf,IAAI,CAAE,KAAK,CACX,MAAM,CAAE,KACZ,CACQ,IAAM,CACV,MAAM,CAAE,CAAC,CACT,QAAQ,CAAE,MACd,CAEA,MAAO,uBAAuB,IAAI,CAAE,CACxB,IAAM,CACV,IAAI,CAAE,KAAK,CACX,MAAM,CAAE,KACZ,CACQ,UAAY,CAChB,IAAI,CAAE,KAAK,CACX,MAAM,CAAE,KACZ,CACQ,SAAW,CACf,IAAI,CAAE,KAAK,CACX,MAAM,CAAE,KACZ,CACQ,IAAM,CACV,gBAAgB,CAAE,KAAK,CACvB,KAAK,CAAE,KACX,CACJ,CACQ,MAAQ,CACZ,MAAM,CAAE,CACZ"}'
};
const Layout = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  $$result.css.add(css);
  return `${slots.default ? slots.default({}) : ``}`;
});
export {
  Layout as default
};
