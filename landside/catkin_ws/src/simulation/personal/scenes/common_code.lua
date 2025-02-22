require "math"


function reset(inInts, inFloats, inString, inBuffer)
    sim.setObjectPosition(hr, -1, initPos)
    sim.setObjectQuaternion(hr, -1, initPos)
    return {}, {}, {}, ''
end

function applyBuoyancy()
    pos = sim.getObjectPosition(hr, -1)

    res, xsizemin = sim.getObjectFloatParameter(hr, 15)
    res, ysizemin = sim.getObjectFloatParameter(hr, 16)
    res, zsizemin = sim.getObjectFloatParameter(hr, 17)
    res, xsizemax = sim.getObjectFloatParameter(hr, 18)
    res, ysizemax = sim.getObjectFloatParameter(hr, 19)
    res, zsizemax = sim.getObjectFloatParameter(hr, 20)
    xsize = xsizemax - xsizemin
    ysize = ysizemax - ysizemin
    zsize = zsizemax - zsizemin

    grav = sim.getArrayParameter(sim.arrayparam_gravity)
    pos[3] = pos[3] - zsize / 2 --fudge due to inconsistency with relative measurements (?)
    if zsize <= (waterlevel - pos[3]) then
        subdepth = zsize
    else
        subdepth = (waterlevel - pos[3])
    end
    fbuoy = xsize * ysize * subdepth * (-1) * grav[3] * p
    if pos[3] > waterlevel then
        fbuoy = 0
        subdepth = 0
    end

    transform = sim.getObjectMatrix(hr, -1)
    res = sim.invertMatrix(transform)
    relbuoy = sim.multiplyVector(transform, { 0, 0, fbuoy })
    relbuoy_mag = math.sqrt(relbuoy[1]^2 + relbuoy[2]^2 + relbuoy[3]^2)
    relbuoy_normalized = {relbuoy[1]/relbuoy_mag * fbuoy, relbuoy[2]/relbuoy_mag * fbuoy, relbuoy[3]/relbuoy_mag * fbuoy}


    v, angv = sim.getVelocity(hr)
    dragforcelin = {calc_dragforcelin(v[1], ysize, subdepth),
                    calc_dragforcelin(v[2], xsize, subdepth),
                    calc_dragforcelin(v[3], xsize, ysize)}
    if pos[3] > waterlevel then
        dragforcelin[3] = 0
    end
    dragforceang = {calc_dragforceang_roll(angv[1], xsize, ysize, subdepth),
                    calc_dragforceang_pitch(angv[2], xsize, ysize, subdepth),
                    calc_dragforceang_yaw(angv[3], xsize, ysize, subdepth)}

    sim.addForceAndTorque(hr, dragforcelin, dragforceang)
    sim.addForce(hr, {0,0,0}, relbuoy_normalized) -- 0,0,0 is center of buoyancy, here
    print(dragforceang[1], dragforceang[2], dragforceang[3])
end

function get_sign(x)
    if (x > 0) then
        return 1
    elseif (x < 0) then
        return -1
    else
        return 0
    end
end

function calc_dragforcelin(linvel, length, depth)
    if dragType ~= 0 then
        return -p * math.abs(linvel ^ 2) * get_sign(linvel) * dragCoef * length * depth
    end
    return -p * math.abs(linvel) * get_sign(linvel) * dragCoef * length * depth
end

function calc_dragforceang_roll(angvel, xsize, ysize, zsize)
--function calc_dragforceang(angvel, length, depth)
    --if quadratic:
    -- -p * angvelocity * angvelocity * x * y * y * y * dragcoef / 12
    -- if linear
    -- -p * angvelocity * x * y * y * dragcoef / 4
    --angdragfudgecoef = 1 -- 0.05
    --if quadratic then
    --    return -p * math.abs(angvel ^ 2) * get_sign(angvel) * dragcoef * length ^ 3 * depth / 12 * angdragfudgecoef
    --end
    --return -p * math.abs(angvel ^ 1) * get_sign(angvel) * dragcoef * length ^ 2 * depth / 4 * angdragfudgecoef
    r0 = (ysize + zsize)/4
    return -p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefroll * math.pi * r0^4 * (0.4*r0 + xsize)
end

function calc_dragforceang_pitch(angvel, xsize, ysize, zsize)
    tau1 = -(1/32) * p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefpitch * (xsize)^4 * zsize
    tau2 = -(1/16) * p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefpitch * (zsize)^3 * xsize * ysize
    tau3 = -(1/16) * p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefpitch * (xsize)^3 * zsize * ysize
    return (2 * tau1) + (2 * tau2) + (2 * tau3)
end

function calc_dragforceang_yaw(angvel, xsize, ysize, zsize)
    tau1 = -(1/32) * p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefyaw * (xsize)^4 * ysize
    tau2 = -(1/16) * p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefyaw * (xsize)^3 * ysize * zsize
    tau3 = -(1/16) * p * math.abs(angvel^2) * get_sign(angvel) * angdragcoefyaw * (ysize)^3 * xsize * zsize
    return (2 * tau1) + (2 * tau2) + (2 * tau3)
end

-- Enables or disables simulation of drag and buoyancy for this object.
-- Params:
--   inInts: Array whose first value is nonzero if buoyancy should be enabled
--   and 0 otherwise
-- Outputs:
--   None
function enableBuoyancyDrag(inInts, inFloats, inString, inBuffer)
    buoyancyEnabled = inInts[1]
    return {},{},{},''
end

-- Sets the drag coefficient for this object.
-- Params:
--   inFloats: Array whose first value is the new drag coefficient.
-- Outputs:
--   None
function setDragCoefficient(inInts, inFloats, inString, inBuffer)
    dragCoef = inFloats[1]
    return {},{},{},''
end

-- Sets the type of drag for this object to linear or quadratic.
-- Params:
--   inInts: Array whose first value 0 if linear and nonzero
--   if quadratic
-- Outputs:
--   None
function setDragType(inInts, inFloats, inString, inBuffer)
    dragType = inInts[1]
    return {},{},{},''
end

-- Sets the mass of this object.
-- Params:
--   inFloats: Array whose first value is the mass of the object.
-- Outputs:
--   None
function setMass(inInts, inFloats, inString, inBuffer)
    sim.setShapeMass(handle, inFloats[1])
    return {},{},{},''
end

-- Adds an anchor point to this object. An anchor point represents a rope
-- which is tied to the object at one end and tied to the floor at the other.
-- Params:
--   TODO: Decide params
-- Outputs:
--   None
function addAnchorPoint(inInts, inFloats, inString, inBuffer)
    return {},{},{},''
end

-- Removes an anchor point from this object. An anchor point represents a rope
-- which is tied to the object at one end and tied to the floor at the other.
-- Params:
--   TODO: Decide params
-- Outputs:
--   None
function removeAnchorPoint(inInts, inFloats, inString, inBuffer)
    return {},{},{},''
end