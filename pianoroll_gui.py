# # # # # # # # # # # # # # # # # # # # # # # # # #
# GUI definition for a pianoroll Optimization app #
# # # # # # # # # # # # # # # # # # # # # # # # # #

import PySimpleGUI as sg

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
from playsound import playsound

class PianorollGUI():

	def get_layout(self,  use_target = False):

		# -----Window for visualizing the generated samples----- #

		canvas_column = [
			[
				sg.Text("Generated sample (1/64)", justification="center",key="sample_text"),
			],
			[
				sg.Canvas(key="-SAMPLE_CANVAS-")
			],
			[
				sg.Button("PLAY")
			]
		]

		# -----Window for evaluating the generated samples----- #

		eval_column = [
			[
				sg.Text("Score", justification="center",key="slider_text")
			],
			[
				sg.Slider(range=(0.0,10.0), default_value = 5.0, resolution=0.1, size=(20,20),key="-SLIDER-"),
				sg.Button("OK")
			],
			[
				sg.InputText(key="-INPUT-",size=(10,2))
			]
		]

		if use_target:

			# -----Window for visualizing the reference----- #
			target_column = [
				[
					sg.Text("Reference sample (---) Score: ---", justification="center",key="ref_text"),
				],
				[
					sg.Canvas(key="-REF_CANVAS-")
				],
				[
					sg.Button("PLAY REFERENCE")
				]
			]

			layout = [
					[
						sg.Column(canvas_column),
						sg.VSeparator(),
						sg.Column(target_column),
						sg.VSeparator(),
						sg.Column(eval_column)
					]
				]
		
		else:

			layout = [
					[
						sg.Column(canvas_column),
						sg.VSeparator(),
						sg.Column(eval_column),
					]
				]

		return layout

	def __init__(self, sample_path, use_ref = False, ref_path="ref.wav"):
		"""
		Intializes the GUI with a canvas to plot the tracks,
		a button to play them and a slider to rate the music.

		sample_path should be the path where the .wav file
		of the generated samples is stored (e.g. "/some/directory/sample.wav")

		if setting a reference, use_ref should be True
		and ref_path should point to the .wav file to be reproduced
		"""

		layout = self.get_layout(use_ref)

		self.sample_path = sample_path

		if use_ref:
			
			self.window = sg.Window("GPianoroll",layout=layout,finalize=True)
			self.reference_fig, self.reference_view = plt.subplots(3,1)
			self.reference_fig.set_figwidth(8)
			self.reference_fig.set_figheight(6)
			self.reference_fig_agg = FigureCanvasTkAgg(self.reference_fig,self.window["-REF_CANVAS-"].TKCanvas)
			self.reference_fig_agg.draw()
			self.reference_fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
			self.reference_path = ref_path
		else:
			self.window = sg.Window("GPianoroll",layout=layout,finalize=True)

		self.use_reference = use_ref

		# -----Helper code for packing a matplotlib figure into the GUI-----

		self.current_fig, self.current_view = plt.subplots(3,1)
		self.current_fig.set_figwidth(8)
		self.current_fig.set_figheight(6)
		self.current_fig_agg = FigureCanvasTkAgg(self.current_fig,self.window["-SAMPLE_CANVAS-"].TKCanvas)
		self.current_fig_agg.draw()
		self.current_fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

	def show_current_sample(self, sample):
		"""
		The function expects sample to be a Multitrack object
		(Hint: Call musegan.py's 'samples_to_multitrack()' on the numpy array)
		"""

		self.current_fig_agg.get_tk_widget().forget()
		plt.figure(self.current_fig.number)
		sample.plot(axs=self.current_view)
		self.current_fig_agg = FigureCanvasTkAgg(self.current_fig,self.window["-SAMPLE_CANVAS-"].TKCanvas)
		self.current_fig_agg.draw()
		self.current_fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

	def play_current_sample(self):
		"""
		The function plays the signal contained in the .wav file
		"""

		playsound(self.sample_path)
		
	def show_target_sample(self, sample):

		self.reference_fig_agg.get_tk_widget().forget()
		plt.figure(self.reference_fig.number)
		sample.plot(axs=self.reference_view)
		self.reference_fig_agg = FigureCanvasTkAgg(self.reference_fig,self.window["-REF_CANVAS-"].TKCanvas)
		self.reference_fig_agg.draw()
		self.reference_fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

	def play_target_sample(self):

		playsound(self.reference_path)

	def wait_for_input(self):

		score = None

		while True:

			event, values = self.window.read()

			if event == "Exit" or event == sg.WIN_CLOSED:
				break
			elif event == "OK":
				if values['-INPUT-']=="":
					score = values["-SLIDER-"]
				else:
					score = float(values["-INPUT-"])
					self.window["-INPUT-"].update("")
				self.window["-SLIDER-"].update(5.0)
				break
			elif event == "PLAY":
				self.play_current_sample()
			elif event == "PLAY REFERENCE":
				self.play_target_sample()

		return score

	def update_sample_text(self,iter):

		self.window["sample_text"].update("Generated sample ("+str(iter)+"/64)")

	def update_ref_text(self,iter,new_val):

		self.window["ref_text"].update("Reference sample ("+str(iter)+") Score: "+(str(new_val)[:3]))

	def close_window(self):

		plt.figure(self.current_fig.number)
		plt.close()
		if(self.use_reference):
			plt.figure(self.reference_fig.number)
			plt.close()
		self.window.close()
