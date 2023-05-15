import kivy
kivy.require('1.11.1') # replace with your current version of Kivy

from kivy.app import App
from kivy.uix.label import Label

class MyApp(App):
    def build(self):
        return Label(text='Hello World')

if __name__ == '__main__':
    MyApp().run()
