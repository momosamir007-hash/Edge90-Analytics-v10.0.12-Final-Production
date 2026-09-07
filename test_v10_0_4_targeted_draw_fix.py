from app import Engine
# Smoke-test the new method without constructing the full application.
class T:
    def __init__(self, dr): self.dr=dr
e=object.__new__(Engine)
h,a=T(.34),T(.32)
hp,dp,ap=e._apply_draw_evidence_preservation(.46,.18,.36,.30,h,a)
assert abs(hp+dp+ap-1.0) < 1e-8
assert dp > .18
# Non-draw-like context should remain unchanged.
h,a=T(.18),T(.18)
x=e._apply_draw_evidence_preservation(.75,.18,.07,.30,h,a)
assert max(abs(x[i]-v) for i,v in enumerate((.75,.18,.07))) < 1e-9
print('V10.0.4 TARGETED DRAW FIX SMOKE PASS')
