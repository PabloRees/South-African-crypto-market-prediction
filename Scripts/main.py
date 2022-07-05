import markdown
import pdfkit


markdown.markdownFromFile(input='Scripts/WriteUp.md',output='/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/WriteUp.html')

pdfkit.from_file('/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/WriteUp.html',
                '/Users/pablo/Desktop/Masters/Data_Science/19119461_Data_Science_Project/WriteUp.pdf')