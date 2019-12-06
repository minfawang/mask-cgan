import React, { Component } from 'react';
import McganApp from './McganApp';
import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton'

const RUN_INFOS = [
  // run_id, left, right
  ['mask_horse2zebra_h128_nres=3_simpled', 'horse', 'zebra'],
  ['mask_monet2photo_h128_nres=3_simpled', 'monet', 'photo'],
  ['mask_vangogh2photo_h128_nres=3_simpled', 'vangogh', 'photo'],
  ['p80mask_vangogh2photo_h128_nres=3_simpled', 'vangogh', 'photo'],
]


const run_state = {
  left: 'horse',
  right: 'zebra',
  runId: 'mask_horse2zebra_h128_nres=3_simpled',
}


function findRunInfo(runId) {
  for (let i = 0; i < RUN_INFOS.length; ++i) {
    const runInfo = RUN_INFOS[i];
    if (runInfo[0] == runId) {
      return runInfo;
    }
  }
}


export default class App extends Component {
  state = {
    runId: RUN_INFOS[0][0],
  }

  switchRun = (runId, e) => {
    this.setState({ runId });
  }

  renderRunDropdown() {
    const dropdownStyle = { display: 'inline-block' };
    
    const runItems = RUN_INFOS.map(runInfo => {
      const [runId, left, right] = runInfo;
      const active = (this.state.runId == runId);
      return <Dropdown.Item key={runId} onSelect={this.switchRun} active={active} eventKey={runId}>{runId}</Dropdown.Item>;
    });
    
    return (
      <div>
        Model: 
        <DropdownButton id="dropdown-basic-button" title={this.state.runId} style={dropdownStyle}>
        {runItems}
      </DropdownButton>
      </div>
    );
  }
  
  render() {
    const { runId } = this.state;
    const [_, left, right] = findRunInfo(runId);
    return (
      <div className="container">
        {this.renderRunDropdown()}
        <br />
        <McganApp key={runId} left={left} right={right} runId={runId} />
      </div>
    );
  }
}