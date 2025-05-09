# from scrapling.fetchers import Fetcher
from scrapling.fetchers import Fetcher, AsyncFetcher, StealthyFetcher, PlayWrightFetcher

# Do HTTP GET request to a web page and create an Adaptor instance
page = Fetcher.get('https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&as_ylo=2021&q=human+AND+protein+AND+disease&btnG=', stealthy_headers=True)
# Get all text content from all HTML tags in the page except the `script` and `style` tags
page.get_all_text(ignore_tags=('script', 'style'))

# Get all quotes elements; any of these methods will return a list of strings directly (TextHandlers)
quotes = page.css('.quote .text::text')  # CSS selector
quotes = page.xpath('//span[@class="text"]/text()')  # XPath
quotes = page.css('.quote').css('.text::text')  # Chained selectors
quotes = [element.text for element in page.css('.quote .text')]  # Slower than bulk query above

# Get the first quote element
quote = page.css_first('.quote')  # same as page.css('.quote').first or page.css('.quote')[0]

# Tired of selectors? Use find_all/find
# Get all 'div' HTML tags that one of its 'class' values is 'quote'
quotes = page.find_all('div', {'class': 'quote'})
# Same as
quotes = page.find_all('div', class_='quote')
quotes = page.find_all(['div'], class_='quote')
quotes = page.find_all(class_='quote')  # and so on...

# Working with elements
quote.html_content  # Get the Inner HTML of this element
quote.prettify()  # Prettified version of Inner HTML above
quote.attrib  # Get that element's attributes
quote.path  # DOM path to element (List of all ancestors from <html> tag till the element itself)
