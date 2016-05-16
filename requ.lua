require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  self.output:clamp(0, math.huge):pow(2)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  input:clamp(0, math.huge)
  self.gradInput:cmul(input):mul(2)
  return self.gradInput
end

