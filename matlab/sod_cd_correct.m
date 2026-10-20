function w = sod_cd_correct(wcd, dt)
%SOD_CD_CORRECT  Invert the central-difference bias w_cd = sin(w dt)/dt (step h: pass h*dt).
w = asin(min(max(wcd * dt, -1), 1)) / dt;
end
