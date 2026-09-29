# reference/

## blur.m

Hansen's `blur.m` — `function [A,b,x] = blur(N,band,sigma)`, header
`Per Christian Hansen, IMM, 11/11/97` — from the Regularization Tools toolbox.
Beck & Teboulle (2009) §5.2 builds its second test image with this function:
`blur.m` *constructs* the picture (two ellipses, a triangle, a cross) rather
than shipping a raster, so the image has to come from this source.

    sha256  6ab03d347375aab26c847d924ff00cd1936d5324be53adcf4409a910f6890ac5
    lines   87

Source: the original DTU distribution (ReguTools.zip, P. C. Hansen) no longer
resolves — its host answers HTTP 300 in a redirect loop, and the MathWorks File
Exchange download 404s. The file above is taken unmodified from the maintained
vendored copy in `north-numerical-computing/anymatrix`:

    https://raw.githubusercontent.com/north-numerical-computing/anymatrix/master/regtools/private/blur.m

It is byte-identical to the copies vendored in `hadiTab/regu`
(`src/blur.m`) and `lijuno/tikregnc` (`regutools/blur.m`), and to the other
GitHub mirrors carrying the same header — all three sha256s above agree, so
this is Hansen's unmodified v4.1.

Verify:

    sha256sum reference/blur.m
