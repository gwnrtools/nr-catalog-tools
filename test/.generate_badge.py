from PIL import Image, ImageDraw, ImageFont
from coverage import coverage

cov = coverage()
cov.load()
total_cov = cov.report()

badge = Image.new("RGB", (140, 60))
font = ImageFont.load_default()

bd = ImageDraw.Draw(im)

bd.text((10, 5), "coverage:", fill=(255, 255, 255), font=font)
bd.rectangle([(80, 0), (150, 20)], fill=(220, 0, 0))
bd.text((90, 5), "{:.0f}%".format(total_cov), fill=(0, 0, 0), font=font)
badge.save('cov_badge.jpg')
