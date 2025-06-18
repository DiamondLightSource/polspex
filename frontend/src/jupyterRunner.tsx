
import { host } from './api';

import React, { useState } from 'react';

const OpenNotebook: React.FC = () => {
    const [notebookUrl, setNotebookUrl] = useState<string>('');

    const openNotebook = async () => {
        try {
            const response = await fetch(host + '/start-notebook/', {
                method: 'POST'
            });
            const data = await response.json();
            setNotebookUrl(data.notebook_url);
            window.open(data.notebook_url, '_blank');
        } catch (error) {
            console.error('Error opening notebook', error);
        }
    };

    return (
        <div>
            <button onClick={openNotebook}>Open Notebook</button>
            {notebookUrl && <p>Notebook URL: <a href={notebookUrl} target="_blank" rel="noopener noreferrer">{notebookUrl}</a></p>}
        </div>
    );
};

export default OpenNotebook;