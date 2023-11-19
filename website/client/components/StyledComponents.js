// components/StyledComponents.js
import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
`;

export const Content = styled.div`
  text-align: center;
`;

export const Confidence = styled.div`
  margin-top: 10px;
`;

export const Progress = styled.progress`
  width: 100%;
  height: 20px;
`;

export const ConfidenceValue = styled.span`
  display: block;
  margin-top: 5px;
  font-size: 1.2em;
`;
