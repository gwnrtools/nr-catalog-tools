from coverage import Coverage

cov = Coverage()
cov.load()

try:
    total_cov = round(cov.report())
except Exception as e:
    print(f"Warning: Could not generate coverage report: {e}")
    total_cov = 0

badge = f"""<svg width="120" height="20" xmlns="http://www.w3.org/2000/svg">
  <rect width="80" height="20" rx="0" ry="5" fill="black" x="0" y="0" />
  <rect width="40" height="20" fill="red" x="80" y="0" />
  <text x="6" y="13" fill="white" font-size="14">Coverage  </text>
  <text x="85" y="14" fill="black" font-size="14">{total_cov}% </text>
</svg>
"""

with open("cov_badge.svg", "w") as svgf:
    svgf.write(badge)
