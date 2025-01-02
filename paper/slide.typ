// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.5.3": *
#import "@preview/lovelace:0.3.0": *
#import "@local/esch:0.0.0": *
#import "@preview/gviz:0.1.0": * // for rendering dot graphs
#import "@preview/finite:0.3.0": automaton // for rendering automata
#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge, shapes
#import "@preview/equate:0.2.1": equate // <- for numbering equations
#import "@preview/plotst:0.2.0": axis, plot, graph_plot, overlay

// #show "llllll": name => box[
// | | | | | |
// ]

#let title = [| | | | | |]
#show: escher-theme.with(
  aspect-ratio: "16-9",
  config-info(author: "Noah Syrkis", date: datetime.today(), title: title),
  config-common(handout: false),   // <- for presentations
)
#show raw.where(lang: "dot-render"): it => render-image(it.text)
#show: equate.with(breakable: true, sub-numbering: false)
#set math.equation(numbering: "(1.1)", supplement: "Eq.")

#show figure.caption: emph

// body /////////////////////////////////////////////////////////////////////////
#cover-slide()


= Since last time
- #cite(<shao2024>, form: "prose") llm that plays StarCraft II.


= Related Work
- #cite(<xu2024>, form: "prose") presents `LLaVA-CoT`, a vision-model that does reasoning. Could be intersting for us.
- #cite(<pan2024>, form: "prose") does multi-agent coordination through LLM.
- Language model for game play
- LLM hybrid

= Future Work
- #cite(<wang2024b>, form: "prose") long range causal reasoning. Causal relationship enhancment, and individual treatment effect. seems like a good recepie for "smater" AI.



- #cite(<zhou2024a>, form: "prose") presents a survey paper on vision-language geo-foundation models (VLGFMs). Given that poor AI performance in llllll, VLGFMs might be what we need to make that part better.
// - #cite(<valevski2024a>, form: "prose") takes the world models paper to the next level, using diffusion as a real time game engine.
// - #cite(<ruoss2024>, form: "prose") does chess without search trees, using direct observation to action.
- #cite(<han2024a>, form: "prose") list challenges and open problems in multi-agent RL.
// - #cite(<park2024>, form: "prose") does a HomoSilicus like thing. Probably not relevant for us.
- #cite(<guo2024>, form: "prose") survey of multi-agent llm.
- #cite(<hu2024a>, form: "prose") survey of llm based game agents.
- #cite(<yim2024>, form: "prose") theory of mind in llms playing a game with imperfect information.
- #cite(<zhang2024a>, form: "prose") shows pretty cool example of forced coordiantion (two llm's have to learn to cook together, or something like that).
- #cite(<zhang2024b>, form: "prose") Survey of strategic reasoning in LLM.
- #cite(<mecattaf2024>, form: "prose") a little less conversation, a little more action (common sense in 3d space).
- #cite(<paglieri2024>, form: "prose") LLMs suck.


= Other relevant papers
- #cite(<chollet2019a>, form: "prose") presents the Abstraction and Reasoning Corpus (ARC) dataset, which has served as a benchmark for intelligent reasoning.
- #cite(<ZhiNengKeXueXueYuan2023>, form: "prose") presents a survey paper, referenciing a bunch of works that are _a)_ relevant for us, and _b)_ might not be "trending" in our part of the AI community.
- #cite(<rivera2024>, form: "prose") write about the risk of using LLLM in military and diplomatic decision making. Not super relevant for us, but it's a good read. They design a novel wargame to assess the risk of using LLMs in military and diplomatic decision making, exploring exalatory behaviors.
- #cite(<silver2016>, form:"prose"), #cite(<silver2017>, form:"prose"), #cite(<silver2018>, form:"prose"), #cite(<dinan2022>, form:"prose"), #cite(<vinyals2019>, form:"prose"), are obviously relevant.





// Bibliography ////////////////////////////////////////////////////////////////
#set align(top)
#show heading.where(level: 1): set heading(numbering: none)
= References <touying:unoutlined>
#bibliography(
  "zotero.bib",
  title: none,
  style: "ieee",
)
