<head>
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
</head>
<body>
<p>Music is a mix of art and science: a way for humans to express themselves, but also a field with heavy maths and physics involved! A musical piece&#39;s impression depends on the physical properties of the sounds that make it, but also on the listener they reach out to. In this project, we propose an evaluation method for Deep Learning generative models that leverages human feedback as our quality measurement for music. We perform Bayesian Optimization over a low-dimensional latent space where we take samples from, map these samples to the high-dimensional latent space of the musical score generator MuseGAN, and ask the final user to evaluate how they like the output of the model. After some BO iterations, our method finds which point of the latent space is more to the subject&#39;s liking, and extracts one final sample. </p>

<h1 id="model">The model</h1>

<p>We built our own custom version of MuseGAN, completely in PyTorch, and featuring Drums, Guitar and Bass. Here are some samples:</p>
<div>
<midi-visualizer type="piano-roll" id="museGANVisualizer" src="MuseGAN_DBG_samples.mid"></midi-visualizer>
<midi-player src="MuseGAN_DBG_samples.mid" sound-font visualizer="#museGANVisualizer" id="museGANPlayer">
</midi-player>
</div>

<h1 id="opt">The results</h1>
<p>We ask users to explore the latent space of the model and give (numerical) scores to the generated (musical) scores!</p>
<p>We start from the middle of the latent space. This is what the initial sample sounds like:</p>

<div>
    <midi-visualizer type="piano-roll" id="midVisualizer" src="mid_sample.mid"></midi-visualizer>
    <midi-player src="mid_sample.mid" sound-font visualizer="#midVisualizer" id="midPlayer">
    </midi-player>
</div>

<hr />
<p style='margin-top:10px'>After that, we start exploring the latent space. We show our volunteers a total of 64 samples. Like these!</p>

<div style="display:flex; flex-wrap:wrap; align-items:flex-end">
<div>
    <midi-visualizer type="piano-roll" id="Visualizer1r" src="sample_rand1.mid"></midi-visualizer>
    <midi-player src="sample_rand1.mid" sound-font visualizer="#Visualizer1r" id="Player1r">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer2r" src="sample_rand2.mid"></midi-visualizer>
    <midi-player src="sample_rand2.mid" sound-font visualizer="#Visualizer2r" id="Player2r">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer3r" src="sample_rand3.mid"></midi-visualizer>
    <midi-player src="sample_rand3.mid" sound-font visualizer="#Visualizer3r" id="Player3r">
    </midi-player>
</div>
</div>

<hr />
<p style='margin-top:10px'>And here are some of our volunteers' favourite pieces!</p>

<div style="display:flex; flex-wrap:wrap; align-items:flex-end">
<div>
    <midi-visualizer type="piano-roll" id="Visualizer1" src="sample_01.mid"></midi-visualizer>
    <midi-player src="sample_01.mid" sound-font visualizer="#Visualizer1" id="Player1">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer2" src="sample_02.mid"></midi-visualizer>
    <midi-player src="sample_02.mid" sound-font visualizer="#Visualizer2" id="Player2">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer3" src="sample_03.mid"></midi-visualizer>
    <midi-player src="sample_03.mid" sound-font visualizer="#Visualizer3" id="Player3">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer4" src="sample_04.mid"></midi-visualizer>
    <midi-player src="sample_04.mid" sound-font visualizer="#Visualizer4" id="Player4">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer5" src="sample_05.mid"></midi-visualizer>
    <midi-player src="sample_05.mid" sound-font visualizer="#Visualizer5" id="Player5">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer6" src="sample_06.mid"></midi-visualizer>
    <midi-player src="sample_06.mid" sound-font visualizer="#Visualizer6" id="Player6">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer7" src="sample_07.mid"></midi-visualizer>
    <midi-player src="sample_07.mid" sound-font visualizer="#Visualizer7" id="Player7">
    </midi-player>
</div>
<div>
    <midi-visualizer type="piano-roll" id="Visualizer8" src="sample_08.mid"></midi-visualizer>
    <midi-player src="sample_08.mid" sound-font visualizer="#Visualizer8" id="Player8">
    </midi-player>
</div>
</div>

</body>

<hr />
<p style='margin-top:10px'>Want to try your hand with optimization? Check our <a href="https://github.com/mikceroese/GPianoroll">GitHub repository</a> and download the code!</p>

<p>Huge thanks to @cifkao for the <a href="https://github.com/cifkao/html-midi-player/">MIDI visualizer</a>.</p>
